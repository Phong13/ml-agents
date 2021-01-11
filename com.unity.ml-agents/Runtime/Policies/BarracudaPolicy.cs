using System;
using Unity.Barracuda;
using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Policies
{
    /// <summary>
    /// Where to perform inference.
    /// </summary>
    public enum InferenceDevice
    {
        /// <summary>
        /// CPU inference
        /// </summary>
        CPU = 0,

        /// <summary>
        /// GPU inference
        /// </summary>
        GPU = 1
    }

    /// <summary>
    /// The Barracuda Policy uses a Barracuda Model to make decisions at
    /// every step. It uses a ModelRunner that is shared across all
    /// Barracuda Policies that use the same model and inference devices.
    /// </summary>
    internal class BarracudaPolicy : IPolicy
    {
        protected ModelRunner m_ModelRunnerPolicy;
        protected ModelRunner m_ModelRunnerValueEst;
        protected ModelRunner m_ModelRunnerPolicyAndValueEst;
        ActionBuffers m_LastActionBuffer;

        int _modelNum;

        int m_AgentId;

        /// <summary>
        /// Sensor shapes for the associated Agents. All Agents must have the same shapes for their Sensors.
        /// </summary>
        List<int[]> m_SensorShapes;
        SpaceType m_SpaceType;

        /// <inheritdoc />
        public BarracudaPolicy(
            ActionSpec actionSpec,
            NNModel modelPolicy,
            NNModel modelValueEst,
            NNModel modelPolicyAndValueEst,
            InferenceDevice inferenceDevice)
        {
            m_ModelRunnerPolicy = Academy.Instance.GetOrCreateModelRunner(modelPolicy, actionSpec, inferenceDevice);
            if (modelValueEst != null)
            {
                m_ModelRunnerValueEst = Academy.Instance.GetOrCreateModelRunner(modelValueEst, actionSpec, inferenceDevice);
            }
            if (modelPolicyAndValueEst != null)
            {
                m_ModelRunnerPolicyAndValueEst = Academy.Instance.GetOrCreateModelRunner(modelPolicyAndValueEst, actionSpec, inferenceDevice);
            }

            actionSpec.CheckNotHybrid();
            m_SpaceType = actionSpec.NumContinuousActions > 0 ? SpaceType.Continuous : SpaceType.Discrete;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors, int modelNum)
        {
            UnityEngine.Debug.Assert(modelNum <= 0 || modelNum <= 2);
            m_AgentId = info.episodeId;
            if (modelNum == 0)
            {
                m_ModelRunnerPolicy?.PutObservations(info, sensors);
                _modelNum = 0;
            }
            else if (modelNum == 1)
            {
                UnityEngine.Debug.Assert(m_ModelRunnerValueEst != null, "Should do inference if no value est model");
                m_ModelRunnerValueEst.PutObservations(info, sensors);
                _modelNum = 1;
            } else if (modelNum == 2)
            {
                UnityEngine.Debug.Assert(m_ModelRunnerPolicyAndValueEst != null, "Should do inference if no value est model");
                m_ModelRunnerPolicyAndValueEst.PutObservations(info, sensors);
                _modelNum = 2;
            } else
            {
                UnityEngine.Debug.Assert(false);
            }
        }

        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            m_AgentId = info.episodeId;
            ModelRunner mr;
            if (_modelNum == 0) mr = m_ModelRunnerPolicy;
            else if (_modelNum == 1) mr = m_ModelRunnerValueEst;
            else mr = m_ModelRunnerPolicyAndValueEst;
            mr?.PutObservations(info, sensors);
            _modelNum = 0;
        }

        /// <inheritdoc />
        public ref readonly ActionBuffers DecideAction()
        {
            ModelRunner mr;
            if (_modelNum == 0) mr = m_ModelRunnerPolicy;
            else if (_modelNum == 1) mr = m_ModelRunnerValueEst;
            else mr = m_ModelRunnerPolicyAndValueEst;

            mr?.DecideBatch();
            float valueEstimate = 0f;
            var actions = mr?.GetAction(m_AgentId, out valueEstimate);
            if (m_SpaceType == SpaceType.Continuous)
            {
                m_LastActionBuffer = new ActionBuffers(actions, Array.Empty<int>(), valueEstimate);
                return ref m_LastActionBuffer;
            }

            m_LastActionBuffer = ActionBuffers.FromDiscreteActions(actions);
            return ref m_LastActionBuffer;
        }

        public void Dispose()
        {
        }
    }
}
