@echo off

rem Define data names
set "data_names=climate_model_crashes california wine_quality_red concrete_compression banknote yacht_hydrodynamics yeast qsar_biodegradation connectionist_bench_sonar"

rem Define miss types
set "miss_types=mcar mar"

rem Loop over data names
for %%d in (%data_names%) do (
    rem Loop over miss types
    for %%m in (%miss_types%) do (

        python ot_main.py --data_name %%d --miss_type %%m
        python notMIWAE_main.py --data_name %%d --miss_type %%m
        python MIWAE_main.py --data_name %%d --miss_type %%m

    )
)