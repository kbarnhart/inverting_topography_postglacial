#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 07:46:29 2018

@author: barnhark
"""

import os
import glob

import numpy as np
import pandas as pd

import xarray as xr

from joblib import Parallel, delayed
from landlab.io import read_esri_ascii

# set paths to result .nc files
ic_path = [
    "work",
    "WVDP_EWG_STUDY3",
    "study3py",
    "prediction",
    "sew",
    "IC_UNCERTAINTY",
    "model_*",
    "*.*.*",
    "run.*",
    "model_*_*.nc",
]
param_path = [
    "work",
    "WVDP_EWG_STUDY3",
    "study3py",
    "prediction",
    "sew",
    "PARAM_2",
    "model_*",
    "*.*.*",
    "run.*",
    "model_*_*.nc",
]

ic_fn = "initial_condition.csv"
if os.path.exists(ic_fn):
    print("loading ic file")
    initial_condition = pd.read_csv(ic_fn, index_col=0, dtype={"model_name": str})
else:
    print("creating ic file")
    initial_condition = pd.DataFrame(
        {"file_name": np.sort(glob.glob(os.path.join(os.path.sep, *ic_path)))}
    )
    initial_condition["model_name"] = (
        initial_condition.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-1])
        .str.split("_")
        .apply(lambda x: x[1])
    )
    initial_condition["run_number"] = (
        initial_condition.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-2])
        .str.split(".")
        .apply(lambda x: x[1])
        .astype(int)
    )
    initial_condition["model_time"] = (
        initial_condition.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-1])
        .str.split("_")
        .apply(lambda x: x[-1][:4])
        .astype(int)
        * 100
    )
    initial_condition["lowering_future"] = (
        initial_condition.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-3])
        .str.split(".")
        .apply(lambda x: x[0])
    )
    initial_condition["climate_future"] = (
        initial_condition.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-3])
        .str.split(".")
        .apply(lambda x: x[2])
    )
    # for interactions.
    initial_condition["climate_lowering"] = initial_condition[
        ["climate_future", "lowering_future"]
    ].apply(lambda x: ".".join(x), axis=1)
    initial_condition["model_climate"] = initial_condition[
        ["model_name", "climate_future"]
    ].apply(lambda x: ".".join(x), axis=1)
    initial_condition["model_lowering"] = initial_condition[
        ["model_name", "lowering_future"]
    ].apply(lambda x: ".".join(x), axis=1)
    initial_condition["model_climate_lowering"] = initial_condition[
        ["model_name", "climate_future", "lowering_future"]
    ].apply(lambda x: ".".join(x), axis=1)
    initial_condition.to_csv(ic_fn)

param_fn = "parameter.csv"

if os.path.exists(param_fn):
    print("loading parameter file")
    parameter = pd.read_csv(param_fn, index_col=0, dtype={"model_name": str})
else:
    print("creating param file")
    parameter = pd.DataFrame(
        {"file_name": np.sort(glob.glob(os.path.join(os.path.sep, *param_path)))}
    )
    parameter["model_name"] = (
        parameter.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-1])
        .str.split("_")
        .apply(lambda x: x[1])
    )
    parameter["run_number"] = (
        parameter.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-2])
        .str.split(".")
        .apply(lambda x: x[1])
        .astype(int)
    )
    parameter["model_time"] = (
        parameter.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-1])
        .str.split("_")
        .apply(lambda x: x[-1][:4])
        .astype(int)
        * 1000
    )
    parameter["lowering_future"] = (
        parameter.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-3])
        .str.split(".")
        .apply(lambda x: x[0])
    )
    parameter["climate_future"] = (
        parameter.file_name.str.split(pat=os.path.sep)
        .apply(lambda x: x[-3])
        .str.split(".")
        .apply(lambda x: x[2])
    )
    parameter["model_climate_lowering"] = parameter[
        ["model_name", "climate_future", "lowering_future"]
    ].apply(lambda x: ".".join(x), axis=1)
    parameter.to_csv(param_fn)

# seems like these take about 20 minutes to make,  give it more power.

# construct model set dictionary
model_sets = {
    "only842": ["842"],
    "all800s": ["800", "802", "804", "808", "810", "840", "842", "A00"],
}

# evaluate only every 1000 years (rather than every 1000)
times = np.arange(0, 10001, 1000)

# set varibles to ignore for faster opening of nc files
ignore_vars = [
    "K_br",
    "bedrock__elevation",
    "cumulative_erosion__depth",
    "depression__depth",
    "drainage_area",
    "effective_drainage_area",
    "erosion__threshold",
    "flow__sink_flag",
    "initial_topographic__elevation",
    "is_pit",
    "rock_till_contact__elevation",
    "sediment__flux",
    "sediment_fill__depth",
    "soil__depth",
    "soil_production__rate",
    "sp_crit_br",
    "substrate__erodibility",
    "subsurface_water__discharge",
    "surface_water__discharge",
    "topographic__steepest_slope",
    "water__unit_flux_in",
]

# create an output file if it doesn't yet exist.
out_folder = "synthesis_netcdfs_with_param_test"
if os.path.exists(out_folder) is False:
    os.mkdir(out_folder)

path = [
    "work",
    "WVDP_EWG_STUDY3",
    "study3py",
    "auxillary_inputs",
    "dems",
    "sew",
    "modern",
    "dem24fil_ext.txt",
]

modern_dem = os.path.join(os.path.sep, *path)

grd, zzm = read_esri_ascii(modern_dem, name="topographic__elevation", halo=1)

array_ID = os.environ["SLURM_ARRAY_TASK_ID"]

# main analysis loop
parallel_inputs = []
for set_key in np.sort(list(model_sets.keys()))[::-1]:  # for model set
    for t in times:  # for recorded times
        if int(np.remainder(t+1, 11000) / 1000) == int(array_ID):
            print(set_key, t, array_ID)
            parallel_inputs.append(
                [
                    set_key,
                    t,
                    initial_condition,
                    ignore_vars,
                    model_sets,
                    zzm.reshape(grd.shape),
                ]
            )

# (set_key, t, cross_model, initial_condition, ignore_vars, used_models, zzm)=parallel_inputs[10]
def average_results(set_key, t, initial_condition, ignore_vars, used_models, zzm):
    print(set_key, t)
    out_name = os.path.join(
        out_folder, set_key + "_synthesis_" + str(t).zfill(5) + ".nc"
    )
    # if a file exists, don't run
    if os.path.exists(out_name):
        run = False
    else:
        run = True

    if run:
        used_models = model_sets[set_key]  # get used models

        # select the correct parts of the PD dataframe for param and ic
        ic_sel = initial_condition[
            initial_condition.model_name.isin(used_models)
            & (initial_condition.model_time == t) & (initial_condition.run_number < 10)
        ]
        pa_sel = parameter[
            parameter.model_name.isin(used_models) & (parameter.model_time == t) & (initial_condition.run_number < 10)
        ]

        # total:
        ds = xr.open_mfdataset(
            ic_sel.file_name,
            concat_dim="nr",
            engine="netcdf4",
            data_vars=["topographic__elevation"],
            drop_variables=ignore_vars,
        )
        # global mean
        global_mean = ds.topographic__elevation.mean(dim="nr").squeeze()
        out_dataset = xr.Dataset({"expected_topographic__elevation": global_mean})

        # expected cumulative erosion.
        expected_cumulative_erosion = out_dataset.expected_topographic__elevation - zzm
        out_dataset.__setitem__(
            "expected_cumulative_erosion__depth",
            (expected_cumulative_erosion.dims, expected_cumulative_erosion.values),
        )

        # total standard deviation.
        topo_total_exp1_std = ds.topographic__elevation.std(dim="nr").squeeze()
        out_dataset.__setitem__(
            "topo_total_exp1_std",
            (topo_total_exp1_std.dims, topo_total_exp1_std.values),
        )

        # across parameters. At present there is only one model/climate/lowering combination.
        # todo this is only correct because there is only one model/lowering combination.
        if set_key == "all800s":

            ds = xr.open_mfdataset(
                pa_sel.file_name,
                concat_dim="nr",
                engine="netcdf4",
                data_vars=["topographic__elevation"],
                drop_variables=ignore_vars,
            )

            global_mean_param = ds.topographic__elevation.mean(dim="nr").squeeze()
            out_dataset.__setitem__(
                "topo_exp2_mean", (global_mean_param.dims, global_mean_param.values)
            )

            global_std_param = ds.topographic__elevation.std(dim="nr").squeeze()
            out_dataset.__setitem__(
                "topo_exp2_std", (global_std_param.dims, global_std_param.values)
            )

            models = pa_sel.model_name.unique()
            param_model_list = []
            param_var_list = []
            param_correlated_var_list = []

            for model in models:
                pa_sel_model = pa_sel[pa_sel.model_name == model]
                ds = xr.open_mfdataset(
                    pa_sel_model.file_name,
                    concat_dim="nr",
                    engine="netcdf4",
                    data_vars=["topographic__elevation"],
                    drop_variables=ignore_vars,
                )
                within_model_var = ds.var(dim="nr").squeeze()
                within_model_mean = ds.mean(dim="nr").squeeze()

                about_global_variance = (within_model_mean - global_mean_param) ** 2

                param_var_list.append(within_model_var)
                param_model_list.append(about_global_variance)

                param_correlated_var_list.append(
                    (within_model_var + about_global_variance) ** 0.5
                )
                # ds.close()

            topo_exp2_model_std = (
                xr.concat(param_model_list, dim="nm")
                .topographic__elevation.mean(dim="nm")
                .squeeze()
                ** 0.5
            )

            topo_exp2_param_independent_std = (
                xr.concat(param_var_list, dim="nm")
                .topographic__elevation.mean(dim="nm")
                .squeeze()
                ** 0.5
            )

            topo_exp2_model_param_correlated_std = (
                xr.concat(param_correlated_var_list, dim="nm")
                .topographic__elevation.mean(dim="nm")
                .squeeze()
            )

            out_dataset.__setitem__(
                "topo_exp2_model_std",
                (topo_exp2_model_std.dims, topo_exp2_model_std.values),
            )

            out_dataset.__setitem__(
                "topo_exp2_param_independent_std",
                (
                    topo_exp2_param_independent_std.dims,
                    topo_exp2_param_independent_std.values,
                ),
            )

            out_dataset.__setitem__(
                "topo_exp2_model_param_correlated_std",
                (
                    topo_exp2_model_param_correlated_std.dims,
                    topo_exp2_model_param_correlated_std.values,
                ),
            )

        else:
            # if there is only one model.
            ds = xr.open_mfdataset(
                pa_sel.file_name,
                concat_dim="nr",
                engine="netcdf4",
                data_vars=["topographic__elevation"],
                drop_variables=ignore_vars,
            )
            topo_exp2_param_independent_std = ds.topographic__elevation.std(
                dim="nr"
            ).squeeze()
            # ds.close()
        out_dataset.__setitem__(
            "topo_exp2_param_independent_std",
            (
                topo_exp2_param_independent_std.dims,
                topo_exp2_param_independent_std.values,
            ),
        )

        # ds.close()

        # a) across model uncertainty is the variability of the model means about
        # the global mean.
        if set_key == "all800s":
            models = ic_sel.model_name.unique()
            model_list = []
            model_means = {}
            for model in models:
                ic_sel_model = ic_sel[ic_sel.model_name == model]
                ds = xr.open_mfdataset(
                    ic_sel_model.file_name,
                    concat_dim="nr",
                    engine="netcdf4",
                    data_vars=["topographic__elevation"],
                    drop_variables=ignore_vars,
                )
                model_mean = ds.mean(dim="nr").squeeze()
                model_means[model] = model_mean
                model_list.append((model_mean - global_mean) ** 2)
                # ds.close()
            topo_model_std = (
                xr.concat(model_list, dim="nm")
                .topographic__elevation.mean(dim="nm")
                .squeeze()
                ** 0.5
            )
            out_dataset.__setitem__(
                "topo_model_std", (topo_model_std.dims, topo_model_std.values)
            )

        # b) across lowering uncertainty
        lowerings = ic_sel.lowering_future.unique()
        lowering_list = []
        lowering_means = {}
        for lowering in lowerings:
            ic_sel_low = ic_sel[ic_sel.lowering_future == lowering]
            ds = xr.open_mfdataset(
                ic_sel_low.file_name,
                concat_dim="nr",
                engine="netcdf4",
                data_vars=["topographic__elevation"],
                drop_variables=ignore_vars,
            )
            lowering_mean = ds.mean(dim="nr").squeeze()
            lowering_means[lowering] = lowering_mean
            lowering_list.append((lowering_mean - global_mean) ** 2)
            # ds.close()
        topo_lower_std = (
            xr.concat(lowering_list, dim="nl")
            .topographic__elevation.mean(dim="nl")
            .squeeze()
            ** 0.5
        )
        out_dataset.__setitem__(
            "topo_lower_std", (topo_lower_std.dims, topo_lower_std.values)
        )

        # c) across climate
        climates = ic_sel.climate_future.unique()
        climate_list = []
        climate_means = {}
        for climate in climates:
            ic_sel_cli = ic_sel[ic_sel.climate_future == climate]
            ds = xr.open_mfdataset(
                ic_sel_cli.file_name,
                concat_dim="nr",
                engine="netcdf4",
                data_vars=["topographic__elevation"],
                drop_variables=ignore_vars,
            )
            climate_mean = ds.mean(dim="nr").squeeze()
            climate_means[climate] = climate_mean
            climate_list.append((climate_mean - global_mean) ** 2)
            # ds.close()
        topo_cli_std = (
            xr.concat(climate_list, dim="nc")
            .topographic__elevation.mean(dim="nc")
            .squeeze()
            ** 0.5
        )
        out_dataset.__setitem__(
            "topo_cli_std", (topo_cli_std.dims, topo_cli_std.values)
        )

        # calculate interaction terms.

        # lowering-climate
        climate_lowerings = ic_sel.climate_lowering.unique()
        climate_lowering_list = []
        climate_lowering_means = {}
        for climate_lowering in climate_lowerings:
            climate, lowering = climate_lowering.split(".")
            ic_sel_cli_low = ic_sel[ic_sel.climate_lowering == climate_lowering]
            ds = xr.open_mfdataset(
                ic_sel_cli_low.file_name,
                concat_dim="nr",
                engine="netcdf4",
                data_vars=["topographic__elevation"],
                drop_variables=ignore_vars,
            )
            climate_lowering_mean = ds.mean(dim="nr").squeeze()

            climate_lowering_means[climate_lowering] = climate_lowering_mean

            climate_lowering_list.append(
                (
                    climate_lowering_mean
                    + global_mean
                    - climate_means[climate]
                    - lowering_means[lowering]
                )
                ** 2
            )
            # ds.close()
        topo_climate_lowering_interaction_std = (
            xr.concat(climate_lowering_list, dim="nc")
            .topographic__elevation.mean(dim="nc")
            .squeeze()
            ** 0.5
        )
        out_dataset.__setitem__(
            "topo_climate_lowering_interaction_std",
            (
                topo_climate_lowering_interaction_std.dims,
                topo_climate_lowering_interaction_std.values,
            ),
        )

        if set_key == "all800s":

            # model-climate
            model_climates = ic_sel.model_climate.unique()
            model_climate_list = []
            model_climate_means = {}
            for model_climate in model_climates:
                model, climate = model_climate.split(".")
                ic_sel_mod_cli = ic_sel[ic_sel.model_climate == model_climate]
                ds = xr.open_mfdataset(
                    ic_sel_mod_cli.file_name,
                    concat_dim="nr",
                    engine="netcdf4",
                    data_vars=["topographic__elevation"],
                    drop_variables=ignore_vars,
                )
                model_climate_mean = ds.mean(dim="nr").squeeze()
                model_climate_means[model_climate] = model_climate_mean
                model_climate_list.append(
                    (
                        model_climate_mean
                        + global_mean
                        - climate_means[climate]
                        - model_means[model]
                    )
                    ** 2
                )
                # ds.close()
            topo_model_climate_interaction_std = (
                xr.concat(model_climate_list, dim="nc")
                .topographic__elevation.mean(dim="nc")
                .squeeze()
                ** 0.5
            )
            out_dataset.__setitem__(
                "topo_model_climate_interaction_std",
                (
                    topo_model_climate_interaction_std.dims,
                    topo_model_climate_interaction_std.values,
                ),
            )

            # model-lowering
            model_lowerings = ic_sel.model_lowering.unique()
            model_lowering_list = []
            model_lowering_means = {}
            for model_lowering in model_lowerings:
                model, lowering = model_lowering.split(".")
                ic_sel_mod_lower = ic_sel[ic_sel.model_lowering == model_lowering]
                ds = xr.open_mfdataset(
                    ic_sel_mod_lower.file_name,
                    concat_dim="nr",
                    engine="netcdf4",
                    data_vars=["topographic__elevation"],
                    drop_variables=ignore_vars,
                )
                model_lowering_mean = ds.mean(dim="nr").squeeze()
                model_lowering_means[model_lowering] = model_lowering_mean

                model_lowering_list.append(
                    (
                        model_lowering_mean
                        + global_mean
                        - lowering_means[lowering]
                        - model_means[model]
                    )
                    ** 2
                )
                # ds.close()
            topo_model_lowering_interaction_std = (
                xr.concat(model_lowering_list, dim="nc")
                .topographic__elevation.mean(dim="nc")
                .squeeze()
                ** 0.5
            )
            out_dataset.__setitem__(
                "topo_model_lowering_interaction_std",
                (
                    topo_model_lowering_interaction_std.dims,
                    topo_model_lowering_interaction_std.values,
                ),
            )

            # model-lowering-climate
            model_climate_lowerings = ic_sel.model_climate_lowering.unique()
            model_climate_lowering_list = []
            for model_climate_lowering in model_climate_lowerings:
                model, climate, lowering = model_climate_lowering.split(".")
                climate_lowering = climate + "." + lowering
                model_climate = model + "." + climate
                model_lowering = model + "." + lowering
                ic_sel_mod_clim_lower = ic_sel[
                    ic_sel.model_climate_lowering == model_climate_lowering
                ]
                ds = xr.open_mfdataset(
                    ic_sel_mod_clim_lower.file_name,
                    concat_dim="nr",
                    engine="netcdf4",
                    data_vars=["topographic__elevation"],
                    drop_variables=ignore_vars,
                )
                model_climate_lowering_mean = ds.mean(dim="nr").squeeze()
                model_climate_lowering_list.append(
                    (
                        model_climate_lowering_mean
                        - global_mean
                        + lowering_means[lowering]
                        + model_means[model]
                        + climate_means[climate]
                        - climate_lowering_means[climate_lowering]
                        - model_climate_means[model_climate]
                        - model_lowering_means[model_lowering]
                    )
                    ** 2
                )
                # ds.close()
            topo_model_climate_lowering_interaction_std = (
                xr.concat(model_climate_lowering_list, dim="nc")
                .topographic__elevation.mean(dim="nc")
                .squeeze()
                ** 0.5
            )
            out_dataset.__setitem__(
                "topo_model_climate_lowering_interaction_std",
                (
                    topo_model_climate_lowering_interaction_std.dims,
                    topo_model_climate_lowering_interaction_std.values,
                ),
            )

        # calculate sum of interaction terms.
        if set_key == "all800s":
            topo_interactions_std = (
                topo_climate_lowering_interaction_std ** 2
                + topo_model_climate_interaction_std ** 2
                + topo_model_lowering_interaction_std ** 2
                + topo_model_climate_lowering_interaction_std ** 2
            ) ** 0.5
        else:
            topo_interactions_std = topo_climate_lowering_interaction_std

        out_dataset.__setitem__(
            "topo_interactions_std",
            (topo_interactions_std.dims, topo_interactions_std.values),
        )

        # initial condition uncertainty is the remaining uncertainty, total - (model, climate, lowering, interactions)
        # also variance when mcl is controlled for. Calculate directly.
        mcls = ic_sel.model_climate_lowering.unique()
        ic_list = []
        for mcl in mcls:
            ic_sel_mcl = ic_sel[ic_sel.model_climate_lowering == mcl]
            ds = xr.open_mfdataset(
                ic_sel_mcl.file_name,
                concat_dim="nr",
                engine="netcdf4",
                data_vars=["topographic__elevation"],
                drop_variables=ignore_vars,
            )
            ic_list.append(ds.var(dim="nr").squeeze())

        topo_ic_std = (
            xr.concat(ic_list, dim="nic")
            .topographic__elevation.mean(dim="nic")
            .squeeze()
            ** 0.5
        )
        out_dataset.__setitem__("topo_ic_std", (topo_ic_std.dims, topo_ic_std.values))

        # one might assess closure by whether variances close between
        #    topo_total_exp1_std
        # and
        # topo_model_std
        # topo_lower_std
        # topo_cli_std
        # topo_interactions_std
        # topo_ic_std

        # and calcuate total star.
        topo_total_star_std = (
            topo_total_exp1_std ** 2 + topo_exp2_param_independent_std ** 2
        ) ** 0.5
        out_dataset.__setitem__(
            "topo_total_star_std",
            (topo_total_star_std.dims, topo_total_star_std.values),
        )
        # save out.
        out_dataset.to_netcdf(out_name, engine="netcdf4", format="NETCDF4")


output = Parallel(n_jobs=1)(
    delayed(average_results)(*inputs) for inputs in parallel_inputs
)
