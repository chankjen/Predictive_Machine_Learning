# Week 9 Model Explainer: Predictive Maintenance

## What the model is doing
This lab simulates an industrial maintenance setting where failures are rare: only 2.0% of equipment logs are failure events. That imbalance creates the accuracy trap. A model can look excellent by mostly saying "normal," while still missing the alerts maintenance teams care about most.

Think of the model like an experienced mechanic listening to an engine. It does not rely on one sound. It weighs vibration, heat, torque, pressure, wear, speed, and operating age together to decide whether a machine deserves attention.

## How the data was prepared
The dataset contains 6,000 sensor records across 120 assets. Sensor readings were normalized with Min-Max Scaling so that large-unit measurements did not overpower smaller-unit signals. The split was leakage-aware: earlier asset groups were used for training, and later unseen asset groups were held out for testing. That keeps future maintenance behavior from sneaking into the training process.

## What changed after handling imbalance
The baseline Random Forest reached 99.0% accuracy, but the important question is failure recall: how many actual failures were caught. Its failure recall was 65.4%. After using class weighting in XGBoost, failure recall was 73.1%, with an average precision score of 0.861. The PR curve is the better view here because failures are rare and false confidence from accuracy is easy.

SMOTE created synthetic failure examples so the Random Forest could see more minority-class patterns. XGBoost class weighting used a different strategy: it made missed failures more costly during training. In operations, that is often the more direct business framing because missing a true failure can mean downtime, safety risk, or expensive emergency repairs.

## What the model found
The SHAP global summary shows which signals most often pushed predictions toward failure across the test fleet. The strongest drivers were typically vibration, process temperature, torque, pressure, and tool wear. That makes practical sense: rising vibration and heat are common early warning signs for mechanical stress.

For the highest-risk test asset, asset 111 at cycle 50, the model estimated a 100.0% failure probability. The local SHAP explanation shows the specific reasons for that alert. The largest contributors were:

| feature | value | shap_contribution |
| --- | --- | --- |
| process_temp | 0.9128 | 3.4027 |
| vibration | 0.9400 | 2.9491 |
| noise | 0.9105 | 1.3960 |
| asset_id | 111.0000 | -0.6916 |
| tool_wear | 0.8984 | 0.5281 |

## Recommended maintenance action
Treat high-risk alerts as triage signals, not automatic shutdown orders. The recommended workflow is:

1. Review the SHAP explanation for the top two or three drivers.
2. Check the physical system tied to those drivers, such as bearing condition for vibration or cooling/lubrication for temperature.
3. Create a maintenance ticket when the same asset shows repeated high-risk readings or when the SHAP drivers match known failure modes.
4. Track technician feedback so future models learn which alerts were useful.

## Limitations
This lab uses a synthetic dataset designed to mirror an AI4I or C-MAPSS style failure problem. A production model would need real sensor history, confirmed failure labels, asset metadata, and review by maintenance experts before it could be used for operational decisions.
