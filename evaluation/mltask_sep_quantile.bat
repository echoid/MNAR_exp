@echo off

rem Define choices for each argument
set "data_names=banknote concrete_compression wine_quality_white wine_quality_red california climate_model_crashes connectionist_bench_sonar qsar_biodegradation yeast yacht_hydrodynamics"
set "miss_types=quantile"
set "model_names=zero mean knn hyper gain XGB mice mf missforest notmiwae miwae tabcsdi ot"

rem Loop through each combination and run the script
for %%a in (%data_names%) do (
    for %%b in (%miss_types%) do (
        for %%c in (%model_names%) do (
            echo Running script with --data_name %%a --miss_type %%b --modelname %%c
            python mltask_sep.py --data_name %%a --miss_type %%b --modelname %%c
        )
    )
)

pause