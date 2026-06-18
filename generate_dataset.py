"""
HOAMS - Step 1: Generate Synthetic Dataset
Run this first: python generate_dataset.py
Output: appointments_dataset.csv (10,000 rows)
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

DEPARTMENTS = {
    0: {"name": "General Medicine", "avg_duration": 8.4,  "noshow_rate": 0.182, "count": 2800},
    1: {"name": "Orthopedics",      "avg_duration": 12.1, "noshow_rate": 0.157, "count": 1900},
    2: {"name": "Pediatrics",       "avg_duration": 9.6,  "noshow_rate": 0.213, "count": 1600},
    3: {"name": "Gynecology",       "avg_duration": 14.2, "noshow_rate": 0.129, "count": 1400},
    4: {"name": "ENT",              "avg_duration": 7.8,  "noshow_rate": 0.194, "count": 1100},
    5: {"name": "Dermatology",      "avg_duration": 10.3, "noshow_rate": 0.168, "count": 1200},
}

records = []
patient_history = {}  # Stores past behavior per patient

print("Generating 10,000 appointment records...")

for dept_id, dept_info in DEPARTMENTS.items():
    for _ in range(dept_info["count"]):

        # ── Patient demographics
        patient_id    = random.randint(1000, 9999)
        patient_age   = int(np.random.normal(38, 18))
        patient_age   = max(1, min(90, patient_age))
        patient_gender = random.choice([0, 1])
        distance_km   = round(abs(np.random.normal(12, 8)), 1)
        is_new_patient = 1 if random.random() < 0.35 else 0

        # ── Appointment metadata
        visit_type = 0 if is_new_patient else random.choice([0, 1])
        lead_time_days = int(abs(np.random.exponential(5)))
        lead_time_days = min(lead_time_days, 30)

        # Random appointment date within a year
        base_date = datetime(2024, 1, 1)
        appt_date = base_date + timedelta(days=random.randint(0, 364))
        day_of_week = appt_date.weekday()
        month = appt_date.month

        # OPD hours: 8 AM to 1 PM
        appointment_hour = random.choices(
            [8, 9, 10, 11, 12, 13],
            weights=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
        )[0]

        # Indian public holidays (approximate count per month)
        is_holiday = 1 if random.random() < 0.04 else 0

        # ── Historical behavioral features
        if patient_id not in patient_history:
            prior_noshow_count = 0
            avg_past_duration  = round(dept_info["avg_duration"] + np.random.normal(0, 2), 1)
            avg_past_duration  = max(3, avg_past_duration)
            appointments_90d   = random.randint(0, 3)
            # New patient — assign a latent reliability score
            patient_history[patient_id] = {
                "noshows": 0, "avg_dur": avg_past_duration,
                "count_90d": appointments_90d,
                "reliability": np.random.beta(5, 2)  # most people are reliable
            }
        else:
            h = patient_history[patient_id]
            prior_noshow_count = h["noshows"]
            avg_past_duration  = round(h["avg_dur"], 1)
            appointments_90d   = h["count_90d"]

        # ── Compute actual consultation duration (ground truth for slot prediction)
        base_dur = dept_info["avg_duration"]

        # Strong, learnable modifiers
        if is_new_patient:          base_dur += 5.0
        if visit_type == 0:         base_dur += 3.0
        if patient_age > 60:        base_dur += 3.5
        if patient_age < 12:        base_dur += 2.5
        if dept_id in [1, 3]:       base_dur += 3.0   # Ortho + Gynae longer
        if dept_id in [4, 0]:       base_dur -= 2.0   # ENT + GenMed shorter
        if appointment_hour >= 12:  base_dur -= 1.5
        if appointments_90d > 5:    base_dur += 1.5
        if avg_past_duration > 15:  base_dur += 2.0

        duration_noise = np.random.normal(0, 1.5)
        actual_duration = max(3, base_dur + duration_noise)

        # ── Slot Duration Class (target variable 1)
        if actual_duration < 10:
            slot_duration_class = 0   # Short
        elif actual_duration < 20:
            slot_duration_class = 1   # Medium
        else:
            slot_duration_class = 2   # Long

        # ── No-Show (target variable 2)
        noshow_prob = dept_info["noshow_rate"]
        # Strong, learnable modifiers
        noshow_prob += prior_noshow_count * 0.10   # past behavior is key
        noshow_prob += lead_time_days * 0.012       # longer wait = more likely to skip
        noshow_prob += distance_km * 0.005          # far away = more likely to skip
        if is_holiday:           noshow_prob += 0.15
        if appointment_hour >= 12: noshow_prob += 0.08
        if day_of_week >= 5:     noshow_prob += 0.10  # weekends higher
        if day_of_week == 0:     noshow_prob -= 0.05  # Monday lower
        if is_new_patient:       noshow_prob += 0.05  # new patients less committed
        if appointments_90d > 3: noshow_prob -= 0.08  # regular attenders reliable
        noshow_prob = min(0.90, max(0.02, noshow_prob))
        no_show = 1 if random.random() < noshow_prob else 0

        # ── Update patient history
        if patient_id not in patient_history:
            patient_history[patient_id] = {"noshows": 0, "avg_dur": actual_duration, "count_90d": 1}
        else:
            h = patient_history[patient_id]
            h["noshows"] += no_show
            h["avg_dur"] = (h["avg_dur"] + actual_duration) / 2
            h["count_90d"] = min(h["count_90d"] + 1, 10)

        records.append({
            "patient_id":          patient_id,
            "patient_age":         patient_age,
            "patient_gender":      patient_gender,
            "distance_km":         distance_km,
            "is_new_patient":      is_new_patient,
            "department_id":       dept_id,
            "department_name":     dept_info["name"],
            "visit_type":          visit_type,
            "appointment_hour":    appointment_hour,
            "day_of_week":         day_of_week,
            "month":               month,
            "lead_time_days":      lead_time_days,
            "is_holiday":          is_holiday,
            "prior_noshow_count":  prior_noshow_count,
            "avg_past_duration":   avg_past_duration,
            "appointments_90d":    appointments_90d,
            "actual_duration_min": round(actual_duration, 1),
            "slot_duration_class": slot_duration_class,
            "no_show":             no_show,
        })

df = pd.DataFrame(records)
df.to_csv("appointments_dataset.csv", index=False)

print(f"\nDataset saved: appointments_dataset.csv")
print(f"Total records : {len(df)}")
print(f"\nSlot Duration Distribution:")
labels = {0: "Short (0-10 min)", 1: "Medium (10-20 min)", 2: "Long (20+ min)"}
for k, v in labels.items():
    count = (df["slot_duration_class"] == k).sum()
    print(f"  {v}: {count} ({count/len(df)*100:.1f}%)")
print(f"\nNo-Show Rate: {df['no_show'].mean()*100:.1f}%")
print(f"\nDepartment breakdown:")
print(df.groupby("department_name")["no_show"].agg(["count","mean"]).rename(columns={"mean":"no_show_rate"}))
