#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import time


# In[2]:


SATELLITE_MASS = 10.0

THRUST = 1e-3

ISP = 1500

G0 = 9.81

DELTA_V_BUDGET = 25.0

PC_LIMIT = 1e-4


# In[3]:


selected_events = [

    11953,
    6792,

    1,
    25,

    5,
    13,

    12

]


# In[4]:


action_names = {

    0: "No manoeuvre",

    1: "+Along-track Small",

    2: "-Along-track Small",

    3: "+Along-track Medium",

    4: "-Along-track Medium",

    5: "+Cross-track Small",

    6: "-Cross-track Small"

}


# In[5]:


action_dv = {

    0: 0.0,

    1: 0.1,
    2: 0.1,

    3: 0.5,
    4: 0.5,

    5: 0.2,
    6: 0.2

}


# In[6]:


def fuel_used(delta_v):

    fuel = SATELLITE_MASS * (

        1 -

        np.exp(

            -delta_v /

            (ISP * G0)

        )

    )

    return fuel


# In[7]:


def burn_time(delta_v):

    acceleration = THRUST / SATELLITE_MASS

    return delta_v / acceleration


# In[8]:


def log10_risk_to_pc(risk):

    return 10 ** float(risk)


# In[9]:


def p3o_candidate(

        event_id,

        action,

        risk_before,

        risk_after=None,

        time_to_tca_days=0.0,

        delta_v_used=0.0,

        keepout_ok=None):

    action = int(action)

    return {

        "source": "PPO/P3O",

        "event_id": event_id,

        "action": action_names[action],

        "delta_v": action_dv[action],

        "delta_v_used": delta_v_used,

        "time_to_tca_s":
            float(time_to_tca_days) * 86400,

        "pc_before":
            log10_risk_to_pc(risk_before),

        "pc_after":
            None
            if risk_after is None
            else log10_risk_to_pc(risk_after),

        "keepout_ok": keepout_ok,

        "valid_input": True

    }


# In[10]:


def rpm_candidate(

        event_id,

        pc_before,

        pc_after,

        delta_v,

        time_to_tca_s,

        delta_v_used=0.0,

        keepout_ok=None,

        valid_input=True):

    return {

        "source": "RPM",

        "event_id": event_id,

        "action": "RPM candidate",

        "delta_v": delta_v,

        "delta_v_used": delta_v_used,

        "time_to_tca_s": time_to_tca_s,

        "pc_before": pc_before,

        "pc_after": pc_after,

        "keepout_ok": keepout_ok,

        "valid_input": valid_input

    }


# In[11]:


def safety_layer(candidate):

    veto_reasons = []

    hold_reasons = []

    if not candidate["valid_input"]:

        veto_reasons.append(
            "invalid_input"
        )

    if (

        candidate["pc_before"] is None

        or candidate["pc_after"] is None

    ):

        hold_reasons.append(
            "post_manoeuvre_risk_not_verified"
        )

    else:

        if candidate["pc_after"] > PC_LIMIT:

            veto_reasons.append(
                "collision_probability_above_limit"
            )

        if candidate["pc_after"] > candidate["pc_before"]:

            veto_reasons.append(
                "collision_risk_increased"
            )

    total_delta_v = (

        candidate["delta_v_used"]

        +

        candidate["delta_v"]

    )

    if total_delta_v > DELTA_V_BUDGET:

        veto_reasons.append(
            "delta_v_budget_exceeded"
        )

    required_burn_time = burn_time(

        candidate["delta_v"]

    )

    if required_burn_time > candidate["time_to_tca_s"]:

        veto_reasons.append(
            "insufficient_time_for_burn"
        )

    if candidate["keepout_ok"] is False:

        veto_reasons.append(
            "keepout_violation"
        )

    elif candidate["keepout_ok"] is None:

        hold_reasons.append(
            "keepout_not_verified"
        )

    if len(veto_reasons) > 0:

        decision = "VETO"

    elif len(hold_reasons) > 0:

        decision = "HOLD"

    else:

        decision = "PASS"

    return {

        "source": candidate["source"],

        "event_id": candidate["event_id"],

        "decision": decision,

        "veto_reasons": veto_reasons,

        "hold_reasons": hold_reasons,

        "delta_v": candidate["delta_v"],

        "fuel_used_g":
            fuel_used(
                candidate["delta_v"]
            ) * 1000,

        "burn_time_s":
            required_burn_time

    }


# In[12]:


test_cases = [

    (
        rpm_candidate(
            9001,
            2e-4,
            5e-5,
            0.1,
            5000,
            keepout_ok=True
        ),
        "PASS"
    ),

    (
        rpm_candidate(
            9002,
            2e-4,
            1.5e-4,
            0.1,
            5000,
            keepout_ok=True
        ),
        "VETO"
    ),

    (
        rpm_candidate(
            9003,
            5e-5,
            8e-5,
            0.1,
            5000,
            keepout_ok=True
        ),
        "VETO"
    ),

    (
        rpm_candidate(
            9004,
            2e-4,
            5e-5,
            1.0,
            50000,
            delta_v_used=24.5,
            keepout_ok=True
        ),
        "VETO"
    ),

    (
        rpm_candidate(
            9005,
            2e-4,
            5e-5,
            0.5,
            1000,
            keepout_ok=True
        ),
        "VETO"
    ),

    (
        rpm_candidate(
            9006,
            2e-4,
            5e-5,
            0.1,
            5000,
            keepout_ok=False
        ),
        "VETO"
    ),

    (
        p3o_candidate(
            9007,
            1,
            -3.5,
            risk_after=None,
            time_to_tca_days=1.0,
            keepout_ok=True
        ),
        "HOLD"
    ),

    (
        rpm_candidate(
            9008,
            2e-4,
            5e-5,
            0.1,
            5000,
            keepout_ok=None
        ),
        "HOLD"
    ),

    (
        rpm_candidate(
            9009,
            5e-5,
            5e-5,
            0.0,
            5000,
            keepout_ok=True
        ),
        "PASS"
    )

]


# In[13]:


passed = 0

for candidate, expected in test_cases:

    result = safety_layer(candidate)

    if result["decision"] == expected:

        passed += 1


print(

    "Unit tests passed:",

    passed,

    "/",

    len(test_cases)

)


# In[14]:


mapped_actions = 0

for action in action_dv:

    candidate = p3o_candidate(

        event_id=5,

        action=action,

        risk_before=-3.5,

        risk_after=None,

        time_to_tca_days=1.0,

        keepout_ok=None

    )

    if candidate["delta_v"] == action_dv[action]:

        mapped_actions += 1


print(

    "PPO/P3O actions mapped:",

    mapped_actions,

    "/",

    len(action_dv)

)


# In[15]:


benchmark_candidate = rpm_candidate(

    9999,

    2e-4,

    5e-5,

    0.1,

    5000,

    keepout_ok=True

)


iterations = 300000

repeats = 7

timings = []


for repeat in range(repeats):

    start = time.perf_counter()

    for i in range(iterations):

        safety_layer(
            benchmark_candidate
        )

    elapsed = time.perf_counter() - start

    timings.append(
        elapsed / iterations
    )


median_time_us = (

    np.median(timings)

    *

    1e6

)


print(

    "Median safety-layer runtime:",

    round(median_time_us, 3),

    "microseconds per call"

)
