Autonomous CubeSat Collision Avoidance Framework
Overview

This repository contains research prototypes developed to investigate autonomous collision avoidance for CubeSat missions.

The project evaluates and compares three complementary approaches:

Recursive Polynomial Method (RPM) for physics-based manoeuvre optimisation.
P3O/PPO + LSTM Framework for autonomous collision-avoidance decision making using reinforcement learning.
Independent Safety Layer for validating manoeuvre recommendations against mission and operational constraints.

Together, these approaches explore how optimisation, machine learning and safety-assurance mechanisms can support future autonomous spacecraft operations.

Project Objectives

The project investigates methods for reducing collision risk while operating within the constraints of a typical CubeSat mission.

Key objectives include:

Reducing collision probability during conjunction events.
Minimising propellant consumption.
Supporting autonomous manoeuvre decision making.
Respecting ΔV and mission constraints.
Providing an independent safety-verification capability.
Evaluating suitability for future onboard deployment.


1. Recursive Polynomial Method (RPM)
Description

The Recursive Polynomial Method (RPM) is a model-based collision avoidance approach that uses orbital mechanics and collision probability calculations to determine a manoeuvre capable of reducing collision risk below a predefined safety threshold.

Unlike reinforcement learning approaches, RPM does not learn from experience. Instead, it computes a collision-avoidance manoeuvre directly from spacecraft state information and uncertainty estimates.

Key Features
Collision probability (PoC) estimation
Clohessy-Wiltshire relative motion modelling
Minimum ΔV manoeuvre optimisation
Fuel consumption estimation
Burn-duration calculations
Mission budget verification
Safety-threshold compliance
Inputs
Relative position
Position uncertainty
Time to closest approach
Orbital parameters
Hard-body radius assumptions
Outputs
Probability of collision before manoeuvre
Probability of collision after manoeuvre
Required ΔV
Fuel consumption
Burn duration
Safety-status assessment
2. P3O/PPO + LSTM Framework
Description

This prototype investigates concepts from the P3O + LSTM autonomous collision avoidance architecture proposed by Mu et al.

The framework models collision avoidance as a sequential decision-making problem using representative conjunction events derived from the ESA Kelvins Collision Avoidance Challenge dataset.

A PPO-based reinforcement learning policy selects collision-avoidance manoeuvres while considering both collision-risk reduction and fuel expenditure.

Methodology
Conjunction Data Messages (CDMs)
                 ↓
      State Representation
                 ↓
        LSTM Temporal Model
                 ↓
         POMDP Framework
                 ↓
           PPO Policy
                 ↓
      Proposed Manoeuvre
                 ↓
          Safety Layer
                 ↓
           Final Action

Key Features
Sequential CDM processing
POMDP formulation
LSTM-based temporal state representation
PPO reinforcement learning policy
CubeSat propulsion constraints
Fuel-aware decision making
Risk-reduction optimisation
Evaluation Metrics
Estimated collision-risk reduction
Fuel consumption
Manoeuvre-selection behaviour
Total cumulative reward
ΔV expenditure
3. Independent Safety Layer
Description

The safety layer operates independently from the manoeuvre-generation method.

Its purpose is to ensure that candidate manoeuvres satisfy mission and operational constraints before being approved for execution.

The safety layer can be used with both RPM-generated and P3O/PPO-generated manoeuvres.

Example Checks
Collision probability thresholds
ΔV-budget compliance
Burn-time feasibility
Keep-out-zone constraints
Manoeuvre validity
Input-data verification
Decision Outcomes
PASS

The manoeuvre satisfies all required constraints and is approved.

HOLD

Additional information or verification is required before a decision can be made.

VETO

The manoeuvre violates one or more safety constraints and is rejected.

CubeSat Assumptions

The prototypes assume a representative CubeSat platform:

Mass                = 10 kg
Thrust              = 1 mN
Specific Impulse    = 1500 s
ΔV Budget           = 25 m/s


These parameters are used when calculating:

Fuel consumption
Burn duration
Manoeuvre feasibility
Mission-budget compliance
Datasets

The repository uses conjunction-event data derived from the:

ESA Kelvins Collision Avoidance Challenge Dataset


Representative low-risk, medium-risk, high-risk and challenge scenarios were selected for evaluation.

Results

The prototypes are evaluated using:

Collision-risk reduction
Probability of collision reduction
Fuel consumption
ΔV requirements
Manoeuvre feasibility
Computational performance
Safety-layer decisions

Generated outputs include:

Risk-reduction plots
Fuel-usage plots
Manoeuvre-selection summaries
Excel result tables
Limitations

These implementations are research prototypes and are not intended for operational flight use.

Current limitations include:

Simplified orbital dynamics assumptions
Analytical risk-reduction approximations
Limited multi-threat modelling
Evaluation using historical conjunction-event data
Absence of hardware-in-the-loop verification
Future Work

Potential future enhancements include:

High-fidelity orbit propagation
Multi-threat conjunction handling
Onboard processor validation
Hardware-in-the-loop testing
Formal safety verification
Flight software integration
Authors

Developed as part of a research project investigating autonomous collision avoidance for CubeSat missions using optimisation, reinforcement learning and safety-assured decision making.

License

This repository is intended for academic and research purposes. Please acknowledge the project and associated report when reusing or extending this work.
