@echo off

rem Define data names
set "data_names=climate_model_crashes california wine_quality_red concrete_compression banknote yacht_hydrodynamics yeast qsar_biodegradation connectionist_bench_sonar wine_quality_white"
rem Define miss types
set "miss_types=mcar mar"

rem Loop over data names
for %%d in (%data_names%) do (
    rem Loop over miss types
    for %%m in (%miss_types%) do (

        python tabcsdi_main.py --data_name %%d --miss_type %%m
    )
)