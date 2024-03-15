@echo off

rem Define data names
set "data_names=climate_model_crashes california wine_quality_red concrete_compression banknote yacht_hydrodynamics yeast qsar_biodegradation connectionist_bench_sonar"

rem Define miss types
set "miss_types=mcar mar"

rem Loop over data names
for %%d in (%data_names%) do (
    rem Loop over miss types
    for %%m in (%miss_types%) do (

        python mice_main.py --data_name %%d --miss_type %%m
        python hyper_main.py --data_name %%d --miss_type %%m
        python knn_main.py --data_name %%d --miss_type %%m
        python mf_main.py --data_name %%d --miss_type %%m
        python missforest_main.py --data_name %%d --miss_type %%m
        python XGB_main.py --data_name %%d --miss_type %%m
    )
)