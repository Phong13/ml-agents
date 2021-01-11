using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine.Profiling;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Inference
{
    internal struct AgentInfoSensorsPair
    {
        public AgentInfo agentInfo;
        public List<ISensor> sensors;
    }

    internal class ModelRunner
    {
        List<AgentInfoSensorsPair> m_Infos = new List<AgentInfoSensorsPair>();
        Dictionary<int, float[]> m_LastActionsReceived = new Dictionary<int, float[]>();
        List<int> m_OrderedAgentsRequestingDecisions = new List<int>();

        ITensorAllocator m_TensorAllocator;
        TensorGenerator m_TensorGenerator;
        TensorApplier m_TensorApplier;

        NNModel m_Model;
        InferenceDevice m_InferenceDevice;
        IWorker m_Engine;
        bool m_Verbose = false;
        string[] m_OutputNames;
        IReadOnlyList<TensorProxy> m_InferenceInputs;
        IReadOnlyList<TensorProxy> m_InferenceOutputs;
        Dictionary<int, List<float>> m_Memories = new Dictionary<int, List<float>>();

        SensorShapeValidator m_SensorShapeValidator = new SensorShapeValidator();

        bool m_VisualObservationsInitialized;

        /// <summary>
        /// Initializes the Brain with the Model that it will use when selecting actions for
        /// the agents
        /// </summary>
        /// <param name="model"> The Barracuda model to load </param>
        /// <param name="actionSpec"> Description of the action spaces for the Agent.</param>
        /// <param name="inferenceDevice"> Inference execution device. CPU is the fastest
        /// option for most of ML Agents models. </param>
        /// <param name="seed"> The seed that will be used to initialize the RandomNormal
        /// and Multinomial objects used when running inference.</param>
        /// <exception cref="UnityAgentsException">Throws an error when the model is null
        /// </exception>
        public ModelRunner(
            NNModel model,
            ActionSpec actionSpec,
            InferenceDevice inferenceDevice = InferenceDevice.CPU,
            int seed = 0)
        {
            Model barracudaModel;
            m_Model = model;
            m_InferenceDevice = inferenceDevice;
            m_TensorAllocator = new TensorCachingAllocator();
            if (model != null)
            {
#if BARRACUDA_VERBOSE
                m_Verbose = true;
#endif

                D.logEnabled = m_Verbose;

                barracudaModel = ModelLoader.Load(model);

                {
                    hasValueEst = barracudaModel.outputs.Contains("value_estimate");
                    hasPolicy = barracudaModel.outputs.Contains("action");
                    hasValueEstOptimizer = barracudaModel.outputs.Contains("optimizer/value_estimate");
                    //UnityEngine.Debug.Log("Created model: " + model.name + " hasValue:" + hasValueEst + " hasPolicy: " + hasPolicy + " hasValueEstOptimizer: " + hasValueEstOptimizer);
                }

                var executionDevice = inferenceDevice == InferenceDevice.GPU
                    ? WorkerFactory.Type.ComputePrecompiled
                    : WorkerFactory.Type.CSharp;
                m_Engine = WorkerFactory.CreateWorker(executionDevice, barracudaModel, m_Verbose);
            }
            else
            {
                barracudaModel = null;
                m_Engine = null;
                hasPolicy = false;
                hasValueEst = false;
                hasValueEstOptimizer = false;
                UnityEngine.Debug.LogError("Should not be creating a Barracuda ModelRunner if model was null ");
            }

            m_InferenceInputs = BarracudaModelParamLoader.GetInputTensors(barracudaModel);
            m_OutputNames = BarracudaModelParamLoader.GetOutputNames(barracudaModel, hasPolicy, hasValueEst, hasValueEstOptimizer);
            m_TensorGenerator = new TensorGenerator(
                seed, m_TensorAllocator, m_Memories, barracudaModel);
            m_TensorApplier = new TensorApplier(
                actionSpec, seed, m_TensorAllocator, m_Memories, barracudaModel);
        }

        static Dictionary<string, Tensor> PrepareBarracudaInputs(IEnumerable<TensorProxy> infInputs)
        {
            var inputs = new Dictionary<string, Tensor>();
            foreach (var inp in infInputs)
            {
                inputs[inp.name] = inp.data;
            }

            return inputs;
        }

        public void Dispose()
        {
            if (m_Engine != null)
                m_Engine.Dispose();
            m_TensorAllocator?.Reset(false);
        }

        List<TensorProxy> FetchBarracudaOutputs(string[] names)
        {
            var outputs = new List<TensorProxy>();
            foreach (var n in names)
            {
                Tensor output;

                try
                {
                    output = m_Engine.PeekOutput(n);
                } catch(System.Exception e)
                {
                    UnityEngine.Debug.LogError("Could not find tensor: " + n + "  " + m_Model.name);
                    throw e;
                }

                if (output != null)
                {
                    //if (n.EndsWith("value_estimate")) UnityEngine.Debug.Log("Adding tensorproxy for: " + n);
                    outputs.Add(TensorUtils.TensorProxyFromBarracuda(output, n));
                } else
                {
                    UnityEngine.Debug.LogError("No TensorProxy added for: " + n + "  " + m_Model.name);
                }
            }

            return outputs;
        }

        public void PutObservations(AgentInfo info, List<ISensor> sensors)
        {
#if DEBUG
            m_SensorShapeValidator.ValidateSensors(sensors);
#endif
            m_Infos.Add(new AgentInfoSensorsPair
            {
                agentInfo = info,
                sensors = sensors
            });

            // We add the episodeId to this list to maintain the order in which the decisions were requested
            m_OrderedAgentsRequestingDecisions.Add(info.episodeId);

            if (!m_LastActionsReceived.ContainsKey(info.episodeId))
            {
                m_LastActionsReceived[info.episodeId] = null;
                m_lastValueEstimate[info.episodeId] = 0f;
            }
            if (info.done)
            {
                // If the agent is done, we remove the key from the last action dictionary since no action
                // should be taken.
                m_LastActionsReceived.Remove(info.episodeId);
                m_lastValueEstimate.Remove(info.episodeId);
            }
        }

        public void DecideBatch()
        {
            var currentBatchSize = m_Infos.Count;
            if (currentBatchSize == 0)
            {
                return;
            }
            if (!m_VisualObservationsInitialized)
            {
                // Just grab the first agent in the collection (any will suffice, really).
                // We check for an empty Collection above, so this will always return successfully.
                var firstInfo = m_Infos[0];
                m_TensorGenerator.InitializeObservations(firstInfo.sensors, m_TensorAllocator);
                m_VisualObservationsInitialized = true;
            }

            Profiler.BeginSample("ModelRunner.DecideAction");

            Profiler.BeginSample($"MLAgents.{m_Model.name}.GenerateTensors");
            // Prepare the input tensors to be feed into the engine
            m_TensorGenerator.GenerateTensors(m_InferenceInputs, currentBatchSize, m_Infos);
            Profiler.EndSample();

            Profiler.BeginSample($"MLAgents.{m_Model.name}.PrepareBarracudaInputs");
            var inputs = PrepareBarracudaInputs(m_InferenceInputs);
            Profiler.EndSample();

            // Execute the Model
            Profiler.BeginSample($"MLAgents.{m_Model.name}.ExecuteGraph");

            //System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            //sw.Start();
            m_Engine.Execute(inputs);
            //sw.Stop();
            //UnityEngine.Debug.Log(sw.Elapsed.TotalMilliseconds.ToString()) ;


            Profiler.EndSample();

            Profiler.BeginSample($"MLAgents.{m_Model.name}.FetchBarracudaOutputs");
            m_InferenceOutputs = FetchBarracudaOutputs(m_OutputNames);
            Profiler.EndSample();

            Profiler.BeginSample($"MLAgents.{m_Model.name}.ApplyTensors");
            // Update the outputs
            m_TensorApplier.ApplyTensors(m_InferenceOutputs, m_OrderedAgentsRequestingDecisions, m_LastActionsReceived);

            {
                // Hack to put the valueEstimate back in
                TensorProxy valueEstimateTensor = null;
                for (int i = 0; i < m_InferenceOutputs.Count; i++)
                {
                    if (hasValueEst && m_InferenceOutputs[i].name.Equals(TensorNames.ValueEstimateOutput))
                    {
                        valueEstimateTensor = m_InferenceOutputs[i];
                    }
                    else if (hasValueEstOptimizer && m_InferenceOutputs[i].name.Equals(ValueEstimateOutputOptimizer))
                    {
                        valueEstimateTensor = m_InferenceOutputs[i];
                    }
                }

                if (hasValueEstOptimizer || hasValueEstOptimizer) UnityEngine.Debug.Assert(valueEstimateTensor != null, "Supposed to have value estimate but doesn't " + m_Model);
                if (valueEstimateTensor != null)
                {
                    UnityEngine.Debug.Assert(m_OrderedAgentsRequestingDecisions.Count == m_Infos.Count, "Should be same");
                    for (int i = 0; i < m_OrderedAgentsRequestingDecisions.Count; i++)
                    {
                        // it is correct to be using i to lookup value in tensor and m_OrderedAgentsRequestingDecisions[i] to cache value in m_lastValueEstimate
                        Tensor d = valueEstimateTensor.data;
                        //UnityEngine.Debug.Log("Reading value count: " + m_Infos.Count + "  idx:" + i + " val: " + valueEstimateTensor.data + 
                        //    "  [" + d.batch + "," + d.width + ", " + d.height + "," + d.channels + "] leng: "+ valueEstimateTensor.shape.Length + "   estimate: " + valueEstimateTensor.data[0, 0, 0, 0]);
                        m_lastValueEstimate[m_OrderedAgentsRequestingDecisions[i]] = valueEstimateTensor.data[i, 0];
                    }
                }
            }

            Profiler.EndSample();

            Profiler.EndSample();

            m_Infos.Clear();

            m_OrderedAgentsRequestingDecisions.Clear();
        }

        public bool HasModel(NNModel other, InferenceDevice otherInferenceDevice)
        {
            return m_Model == other && m_InferenceDevice == otherInferenceDevice;
        }

        public float[] GetAction(int agentId)
        {
            if (m_LastActionsReceived.ContainsKey(agentId))
            {
                return m_LastActionsReceived[agentId];
            }
            return null;
        }

        Dictionary<int, float> m_lastValueEstimate = new Dictionary<int, float>();
        public float[] GetAction(int agentId, out float valueEstimate)
        {
            if (m_LastActionsReceived.ContainsKey(agentId))
            {
                valueEstimate = m_lastValueEstimate[agentId];
                return m_LastActionsReceived[agentId];
            }

            valueEstimate = 0f;
            return null;
        }

        bool hasPolicy;
        bool hasValueEst;
        bool hasValueEstOptimizer;

        public const string ValueEstimateOutputOptimizer = "optimizer/value_estimate";
    }
}
