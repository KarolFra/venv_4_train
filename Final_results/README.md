# Selected results (no weights)

This folder holds small, shareable artifacts for two runs:
- `pcbb_v5i_yolo11s_e180`
- `KICADgood_v5i_yolo11s_e180`

Included: key metric plots (results, curves, confusion matrices), sample predictions (`val_batch0_pred.jpg` / `val_batch0_labels.jpg`), and representative label visualizations. Weights and bulk datasets are purposely excluded.

Quick takeaway:
- Both runs show limited real-world generalization because the dataset is tiny (61 images) and dominated by synthetic KiCad screenshots.
- The KICADgood variant especially overfits to synthetic layouts; real-world validation samples are too few for the model to learn robust features.
- Recommendation: collect substantially more real PCB photos, balance classes, and reserve a real-world validation split before further training.

Additional assets:
- experiment_artifacts/ModelPhooto/: mixed design/reference shots (dashboards, schematics, PCB photos)
- experiment_artifacts/Results/: presentation renders and LED test photos
