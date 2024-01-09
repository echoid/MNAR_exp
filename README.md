# MNAR_exp


1. Run ```create_data.ipynb```: load raw data, do the normalization, and do the datasplit of each data.
   datalist : ["banknote","concrete_compression",
            "wine_quality_white","wine_quality_red",
            "california","climate_model_crashes",
            "connectionist_bench_sonar","qsar_biodegradation",
            "yeast","yacht_hydrodynamics"
            ]

2. Run```create_missing.ipynb```
   Create Missing Masks, include missing rate