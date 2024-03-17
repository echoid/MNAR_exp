@echo off

rem Define data names
set "data_names=yeast yacht_hydrodynamics banknote concrete_compression"
rem Define miss types
set "miss_types=mcar mar"

rem Loop over data names
for %%d in (%data_names%) do (
    rem Loop over miss types
    for %%m in (%miss_types%) do (

        python tabcsdi_main.py --data_name %%d --miss_type %%m

    )
)