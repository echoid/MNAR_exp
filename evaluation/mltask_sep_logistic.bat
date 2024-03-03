@echo off

rem Define choices for each argument
set "data_names=california"
set "miss_types=quantile"
set "model_names=ot tabcsdi miwae notmiwae"

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