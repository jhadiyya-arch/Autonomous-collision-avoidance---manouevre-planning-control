#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # PPO-LSTM Collision Avoidance Framework
# 
# Dataset:
# ESA Kelvins Collision Avoidance Challenge
# 
# Method:
# POMDP + LSTM State Representation + PPO Policy + Safety Layer
# 
# Satellite Assumptions:
# 10 kg CubeSat
# 1 mN Thruster
# Isp = 1500 s
# ΔV Budget = 25 m/s

# Kelvins CDMs
#         ↓
# Observation Space
#         ↓
# LSTM Temporal Encoder
#         ↓
# POMDP Belief State
#         ↓
# PPO Policy
#         ↓
# Safety Layer
#         ↓
# Collision Avoidance Action
#         ↓
# Risk & Fuel Evaluation

# In[2]:


df = pd.read_csv(r"C:\Users\User\Downloads\train_data.csv")

print(df.shape)
df.head()


# In[3]:


state_columns = [

    "time_to_tca",
    "risk",
    "miss_distance",
    "relative_speed",

    "relative_position_r",
    "relative_position_t",
    "relative_position_n",

    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",

    "mahalanobis_distance"

]


# In[4]:


plt.figure(figsize=(8,5))

event5 = (
    df[df["event_id"] == 5]
    .sort_values(
        "time_to_tca",
        ascending=False
    )
)
plt.plot(
    event5["time_to_tca"],
    event5["risk"],
    marker="o"
)

plt.gca().invert_xaxis()

plt.xlabel("Time to TCA (days)")
plt.ylabel("Risk")

plt.title("Event 5 Risk Evolution")

plt.grid(True)

plt.savefig(
    "risk_evolution_event5.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[5]:


def get_event_sequence(event_id, dataframe):

    event = dataframe[
        dataframe["event_id"] == event_id
    ]

    event = event.sort_values(
        by="time_to_tca",
        ascending=False
    )

    return event[state_columns]


# In[6]:


from sklearn.preprocessing import MinMaxScaler


# In[7]:


scaler = MinMaxScaler()

scaler.fit(
    df[state_columns]
)

print("Scaler fitted successfully")


# In[8]:


import torch


# In[9]:


import torch.nn as nn


# In[10]:


class CDMLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=11,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            32
        )

    def forward(self,x):

        output,(hidden,cell)=self.lstm(x)

        last_hidden=hidden[-1]

        state_embedding=self.fc(last_hidden)

        return state_embedding


# In[11]:


actions = {

    0 : "No manoeuvre",

    1 : "+Along-track burn",

    2 : "-Along-track burn",

    3 : "+Radial burn",

    4 : "-Radial burn"

}


# In[12]:


class P3ONetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=11,
            hidden_size=64,
            batch_first=True
        )

        self.policy = nn.Linear(
            64,
            7
        )

    def forward(self,x):

        output,(hidden,cell)=self.lstm(x)

        logits = self.policy(
            hidden[-1]
        )

        return logits


# In[13]:


selected_events = [

    11953,
    6792,

    1,
    25,

    5,
    13,

    12,
    21
]


# In[14]:


action_names = {
    0: "No manoeuvre",
    1: "+Along-track Small",
    2: "-Along-track Small",
    3: "+Along-track Medium",
    4: "-Along-track Medium",
    5: "+Cross-track Small",
    6: "-Cross-track Small"
}


# # POMDP Formulation
# 
# Observation:
# CDM state vectors derived from Kelvins conjunction data.
# 
# Actions:
# Discrete along-track and cross-track collision avoidance manoeuvres.
# 
# Transition Model:
# Analytical risk update based on manoeuvre type.
# 
# Reward:
# Risk reduction, miss-distance improvement, and fuel conservation.
# 
# Policy:
# PPO reinforcement learning agent.
# 
# Safety Layer:
# ΔV constraint enforcement.

# In[15]:


def reward_function(
        risk,
        miss_distance,
        action):

    action = int(action)

    new_risk = estimated_new_risk(
        risk,
        action
    )

    reward = 0

    # strongly reward risk reduction
    reward += (
        abs(risk)
        - abs(new_risk)
    ) * 500

    # small miss distance reward
    reward += miss_distance / 10000

    # fuel cost
    reward -= action_dv[action] * 10

    # penalize doing nothing for dangerous events
    if action == 0 and abs(risk) > 20:
        reward -= 50

    return reward


# In[16]:


action_dv = {

    0: 0.0,

    1: 0.1,
    2: 0.1,

    3: 0.5,
    4: 0.5,

    5: 0.2,
    6: 0.2
}


# In[17]:


def safety_layer(action):

    action = int(action)

    # prevent excessive burns

    if action_dv[action] > 0.5:
        return 0

    return action


# In[18]:


risk_reduction = {

    0: 1.00,

    1: 0.95,
    2: 0.95,

    3: 0.80,
    4: 0.80,

    5: 0.85,
    6: 0.85
}


# In[19]:


def estimated_new_risk(risk, action):

    action = int(action)

    return (
        risk *
        risk_reduction[action]
    )


# In[20]:


# 6U/12U CubeSat baseline

SATELLITE_MASS = 10.0      # kg

THRUST = 1e-3              # N (1 mN)

ISP = 1500                 # s

G0 = 9.81                  # m/s²

DELTA_V_BUDGET = 25.0      # m/s


# In[21]:


import numpy as np


# In[22]:


def fuel_used(delta_v):

    fuel = SATELLITE_MASS * (

        1 -
        np.exp(
            -delta_v /
            (ISP * G0)
        )

    )

    return fuel


# In[23]:


from stable_baselines3 import PPO


# In[24]:


training_events = [

    11953,
    6792,

    1,
    25,

    5,
    13,

    12,
    21
]


# In[25]:


import gymnasium as gym
from gymnasium import spaces


# # PPO Training
# 
# The PPO agent is trained on Kelvins conjunction events using the CubeSat collision avoidance environment. The learned policy selects manoeuvres that balance collision-risk reduction and fuel consumption while respecting safety constraints.

# In[26]:


class CubeSatAvoidanceGym(gym.Env):

    def __init__(self, df, event_id):

        super().__init__()

        self.df = df

        self.event_id = event_id

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        self.event = (
            self.df[
                self.df["event_id"] == self.event_id
            ]
            .sort_values(
                "time_to_tca",
                ascending=False
            )
            .reset_index(drop=True)
        )

        self.step_idx = 0

        observation = self.event.loc[
            self.step_idx,
            state_columns
        ].values.astype(np.float32)

        return observation, {}

    def step(self, action):

        action = int(action)
        current = self.event.iloc[
            self.step_idx
        ]

        reward = reward_function(
            current["risk"],
            current["miss_distance"],
            action
        )

        self.step_idx += 1

        terminated = (
            self.step_idx >= len(self.event)
        )

        if terminated:

            observation = np.zeros(
                11,
                dtype=np.float32
            )

        else:

            observation = (
                self.event.loc[
                    self.step_idx,
                    state_columns
                ]
                .values
                .astype(np.float32)
            )

        return (
            observation,
            reward,
            terminated,
            False,
            {}
        )


# plt.figure(figsize=(10,5))
# 
# plt.bar(
#     results_df["event_id"].astype(str),
#     results_df["delta_v"]
# )
# 
# plt.title(
#     "Selected Delta-V by Event"
# )
# 
# plt.ylabel(
#     "Delta-V (m/s)"
# )
# 
# plt.tight_layout()
# 
# plt.savefig(
#     "delta_v_by_event.png"
# )
# 
# plt.show()

# In[27]:


env = CubeSatAvoidanceGym(df, 5)

ppo_model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

ppo_model.learn(
    total_timesteps=50000
)


# In[28]:


ppo_model.save(
    "ppo_event5_v2"
)


# In[29]:


def evaluate_trained_ppo(event_id):

    env = CubeSatAvoidanceGym(
        df,
        event_id
    )

    obs, info = env.reset()

    done = False

    total_reward = 0

    chosen_actions = []

    while not done:

        action, _ = ppo_model.predict(
            obs,
            deterministic=True
        )

        action = safety_layer(action)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        chosen_actions.append(
            int(action)
        )

        total_reward += reward

    return total_reward, chosen_actions


# In[30]:


ppo_results = []

for event_id in selected_events:

    reward, actions = evaluate_trained_ppo(
        event_id
    )

    ppo_results.append({

        "event_id": event_id,

        "total_reward": reward,

        "actions_taken": str(actions),

        "num_actions": len(actions)

    })

ppo_df = pd.DataFrame(
    ppo_results
)

ppo_df


# In[31]:


plt.figure(figsize=(10,5))

plt.bar(
    ppo_df["event_id"].astype(str),
    ppo_df["total_reward"]
)

plt.title(
    "Total PPO Reward by Event"
)

plt.xlabel(
    "Event ID"
)

plt.ylabel(
    "Total Reward"
)

plt.tight_layout()


plt.savefig(
    "ppo_reward_by_event.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[32]:


ppo_metrics = []

for event_id in selected_events:

    reward, actions = evaluate_trained_ppo(
        event_id
    )

    event = df[
        df["event_id"] == event_id
    ]

    risk_before = event["risk"].min()

    dominant_action = max(
        set(actions),
        key=actions.count
    )

    risk_after = estimated_new_risk(
        risk_before,
        dominant_action
    )

    dv_used = (
        action_dv[dominant_action]
        * len(actions)
    )

    fuel = fuel_used(dv_used)

    ppo_metrics.append({

        "event_id": event_id,

        "risk_before": risk_before,

        "risk_after": risk_after,

        "risk_reduction":
            abs(risk_before)
            - abs(risk_after),

        "dominant_action":
            action_names[dominant_action],

        "total_dv":
            dv_used,

        "fuel_used_g":
            fuel * 1000,

        "total_reward":
            reward
    })


# In[33]:


ppo_metrics_df = pd.DataFrame(
    ppo_metrics
)

ppo_metrics_df


# In[34]:


plt.figure(figsize=(10,5))

plt.bar(
    ppo_metrics_df["event_id"].astype(str),
    ppo_metrics_df["risk_reduction"]
)

plt.title(
    "Estimated Risk Reduction by Event"
)

plt.xlabel("Event ID")

plt.ylabel("Risk Reduction")

plt.tight_layout()

plt.savefig(
    "risk_reduction.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[35]:


plt.figure(figsize=(10,5))

plt.bar(
    ppo_metrics_df["event_id"].astype(str),
    ppo_metrics_df["fuel_used_g"]
)

plt.title(
    "Estimated Fuel Usage by Event"
)

plt.xlabel("Event ID")

plt.ylabel("Fuel Used (g)")

plt.tight_layout()

plt.savefig(
    "fuel_usage.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[36]:


plt.figure(figsize=(10,5))

x = np.arange(len(ppo_metrics_df))

width = 0.4

plt.bar(
    x - width/2,
    ppo_metrics_df["risk_before"],
    width,
    label="Before"
)

plt.bar(
    x + width/2,
    ppo_metrics_df["risk_after"],
    width,
    label="After"
)

plt.xticks(
    x,
    ppo_metrics_df["event_id"].astype(str),
    rotation=45
)

plt.title(
    "Collision Risk Before and After PPO Decision"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "risk_before_after.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[37]:


ppo_summary = []

for event_id in selected_events:

    reward, actions = evaluate_trained_ppo(
        event_id
    )

    most_common_action = max(
        set(actions),
        key=actions.count
    )

    ppo_summary.append({

        "event_id": event_id,

        "reward": reward,

        "dominant_action":
            action_names[most_common_action],

        "num_actions":
            len(actions)

    })

ppo_summary_df = pd.DataFrame(
    ppo_summary
)

ppo_summary_df


# In[38]:


from collections import Counter

all_actions = []

for _, row in ppo_df.iterrows():

    actions = eval(
        row["actions_taken"]
    )

    all_actions.extend(actions)

counts = Counter(all_actions)


# In[39]:


plt.figure(figsize=(8,5))

plt.bar(

    [action_names[a]
     for a in counts.keys()],

    counts.values()

)

plt.title(
    "PPO Action Selection Frequency"
)

plt.ylabel(
    "Count"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "action_frequency.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# In[40]:


with pd.ExcelWriter(
    "final_results.xlsx"
) as writer:

    ppo_metrics_df.to_excel(
        writer,
        sheet_name="PPO Metrics",
        index=False
    )

    ppo_df.to_excel(
        writer,
        sheet_name="PPO Results",
        index=False
    )


    ppo_summary_df.to_excel(
    writer,
    sheet_name="PPO Summary",
    index=False
    )


# In[41]:


print(ppo_metrics_df)


# In[ ]:




