"""YOLO26 item-localization stage (detect-then-classify pipeline).

Separate from ``edgewaste.data`` because the detection dataset (TACO, in YOLO
box-annotation format) is a fundamentally different shape than the
folder-per-class classification datasets in ``edgewaste.taxonomy`` — it can't
go through ``edgewaste-ingest``.
"""
