rem Define choices for each argument
set "data_names=california"
set "miss_types=diffuse"
set "model_names=tabcsdi"

rem Loop through each combination and run the script
for %%a in (%data_names%) do (
    for %%b in (%miss_types%) do (
        for %%c in (%model_names%) do (
            echo Running script with --data_name %%a --miss_type %%b --modelname %%c
            python mltask.py --data_name %%a --miss_type %%b --modelname %%c
        )
    )
)

pause