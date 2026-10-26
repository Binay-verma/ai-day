---
title: Explainable Edge AI Copilot
emoji: 🚨
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Explainable AI for falls, fire/smoke and crowd emergencies
---

# Explainable Edge AI Copilot for Smart Emergency Response (team local maxima)

Public demo for SoCSE AI Day 2026: a recorded demo reel loops on one "camera" and the copilot detects
falls, smoke / fire and abnormal crowd movement live, explaining every alert (exact evidence ledger,
counterfactual, What-if, verified text) and recommending actions from campus playbooks.

**Open the app directly at its `*.hf.space` address** (the Space page frame blocks the login cookie).
Sign in as `judge` with the password given by the team.

Try: wait for the fall (~27 s into the reel), fire (~47 s) and crowd (~66 s) alerts; press **Site scenario**
(smoke at the East stairwell, crowd redirected); type `falls today` in the search box; open **What if?** on
an alert; download **Evidence bundle (signed)**.

Decision support only: it supplements, and never replaces, the certified fire alarm system and emergency
services. Footage: GMDCSA24 (MIT), FURG fire (CC0), UMN crowd (research use). Model weights derive from
Ultralytics YOLO11 (AGPL-3.0); the source of this service is in the Files tab.
