@echo off

rem Define data names
set "data_names=connectionist_bench_sonar wine_quality_white qsar_biodegradation"

rem Define miss types
set "miss_types=mcar mar"

rem Loop over data names
for %%d in (%data_names%) do (
    rem Loop over miss types
    for %%m in (%miss_types%) do (

        python tabcsdi_main.py --data_name %%d --miss_type %%m

    )
)