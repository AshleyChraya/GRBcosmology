import argparse  # Commandline input
from collections import OrderedDict as odict

import getdist.plots as gdplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
from scipy.integrate import quad
from scipy.interpolate import interp1d
from mpi4py import MPI
from cobaya.log import LoggedError

directory = "/media/ash/1tb/github/Quasar-Cosmology/Ashley/"
# Commandline input with Quasarstable.txt as default value
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-DataQuasar",
    type=str,
    default="Quasarstable.txt",
    help="Put either Quasarstable.txt or QSO_2065.txt",
)
parser.add_argument(
    "-Data",
    type=str,
    default="quasar",
    help="Put quasar for only quasars or both for SN+Quasar",
)
parser.add_argument(
    "-Calibration",
    type=str,
    default="calib",
    help="Put calib for calibration and nocalib for no calibration",
)
parser.add_argument(
    "-Equation",
    type=str,
    default="lum",
    help="Put lum for Luminosity relation, dm for Distance modulus relation",
)
parser.add_argument(
    "-Evolution",
    type=str,
    default="noEvol",
    help="Put noEvol for No Evolution, fixedEvol for fixed Evolution, functionEvol for Evolutionary functions",
)
parser.add_argument(
    "-Parameters",
    type=str,
    default="gbsv",
    help="Put gbsv for varying non cosmological fit, Om for varying only Om, H0 for varying only H0, OmH0 for varying both, gbsvOm for varying g b sv and Om, and similarly put gbsvH0 and gbsvOmH0 for corresponding variations ",
)

args = parser.parse_args()


# Reading quasars data
def QSO_data_read(data, O_m, H0):

    QSOdata = pd.read_csv(f"tables_and_data/{str(data)}", sep="\t")
    z = QSOdata["z"].to_numpy()
    logFluxUV = QSOdata["logFluxUV"].to_numpy()  # Units of flux are erg s^-1 cm^-2
    logFluxX = QSOdata["logFluxX"].to_numpy()
    logFluxUVerr = QSOdata["logFluxErrUV"].to_numpy()
    logFluxXerr = QSOdata["logFluxErrX"].to_numpy()

    d_mpc, d_cm = distance_luminosity_Q(O_m, H0)

    # For no evolution
    if args.Evolution == "noEvol":
        # unit of Luminosity is erg /s # Flux units erg s^-1 cm^-2, hence d_cm is used
        logLumUV = logFluxUV + np.log10(4 * np.pi * d_cm**2)
        logLumX = logFluxX + np.log10(4 * np.pi * d_cm**2)
        logLumUVerr = logFluxUVerr
        logLumXerr = logFluxXerr

    # For fixed evolution
    elif args.Evolution == "fixedEvol":

        kLUV = 4.36
        kLUVerr = 0.08
        kLX = 3.36
        kLXerr = 0.07

        logLumUV = logFluxUV + np.log10(4 * np.pi * d_cm**2) - kLUV * np.log10(1 + z)  # Fuv = Fuv / (1+z)**KLuv
        logLumX = logFluxX + np.log10(4 * np.pi * d_cm**2) - kLX * np.log10(1 + z)  # Fx = Fx / (1+z)**KLx
        logLumUVerr = np.sqrt(logFluxUVerr**2 + kLUVerr**2 * np.log10(1 + z) ** 2)
        logLumXerr = np.sqrt(logFluxXerr**2 + kLXerr**2 * np.log10(1 + z) ** 2)

    # For Evolutionary function
    elif args.Evolution == "functionEvol":
        QSO2data = pd.read_csv("tables_and_data/kresultsLXvsOm.txt", sep="\t")

        Om_k = QSO2data["Om"].to_numpy()
        kLX1 = QSO2data["kLX"].to_numpy()
        kLX1_err = (QSO2data["kLXMax"].to_numpy() - QSO2data["kLXMin"].to_numpy()) / 2

        kLX_in = interp1d(
            Om_k, kLX1, kind="cubic"
        )  # interpolation of KLx vs Om with cubic function...This is a func of O_m

        kLXerr_in = interp1d(Om_k, kLX1_err, kind="cubic")  # interpolation of KLxerr vs Om...how error is interpolated?

        # kLUV

        QSO3data = pd.read_csv("tables_and_data/kresultsLUVvsOm.txt", sep="\t")

        Om_k1 = QSO3data["Om"].to_numpy()
        kLUV1 = QSO3data["kLUV"].to_numpy()
        kLUV1_err = (QSO3data["kLUVMax"].to_numpy() - QSO3data["kLUVMin"].to_numpy()) / 2

        kLUV_in = interp1d(Om_k1, kLUV1, kind="cubic")

        kLUVerr_in = interp1d(Om_k1, kLUV1_err, kind="cubic")

        logLumUV = (
            logFluxUV + np.log10(4 * np.pi * d_cm**2) - kLUV_in(O_m) * np.log10(1 + z)
        )  # Fuv = Fuv / (1+z)**KLuv where KLuc is func of Om
        logLumX = logFluxX + np.log10(4 * np.pi * d_cm**2) - kLX_in(O_m) * np.log10(1 + z)  # Fx = Fx / (1+z)**KLx
        logLumUVerr = np.sqrt(logFluxUVerr**2 + kLUVerr_in(O_m) ** 2 * np.log10(1 + z) ** 2)
        logLumXerr = np.sqrt(logFluxXerr**2 + kLXerr_in(O_m) ** 2 * np.log10(1 + z) ** 2)

    return z, logFluxUV, logFluxUVerr, logFluxX, logFluxXerr, logLumUV, logLumX, logLumUVerr, logLumXerr


# DL in cm and Mpc units for Quasars
def distance_luminosity_Q(O_m, H0):

    QSOdata = pd.read_csv(f"tables_and_data/{str(data)}", sep="\t")
    zQ = QSOdata["z"].to_numpy()
    O_l = 1 - O_m
    coefQ = (ckm / H0) * (1 + zQ)  # Units of this is Mpc

    d_par = np.array([])
    for i in zQ:
        Integ = quad(integrand, 0, i, args=(O_m, O_l))
        d_par = np.append(d_par, Integ[0])

    d_mpc = np.array(coefQ * d_par)  # Distance luminosity # Unit of it is Mpc
    d_cm = d_mpc * conversionfrom_cm_to_Mpc  # As flux unit is [erg s^(-1) cm^(-2)], 1 cm = 1/ (3.0857 *10^24) Mpc

    return d_mpc, d_cm


# Reading SN data
def SN_data_read(O_m, H0):

    # Covariance data
    InverseCmat = pd.read_csv(
        "tables_and_data/AVGCfrom1to1048ORD.txt", sep="\t", header=None
    )  # import the matrices from SNe (Cmat) where C is the complete Coariance matrix from Dstat and Csys already inverted
    # Cov = Dstat + Csyst;  statistical matrix Dstat only has the diagonal components, it includes the distance error of each SNIa
    # Csys is the systematic covariance for each SNIa
    Cinverse = np.array(InverseCmat)
    Cmat = np.linalg.inv(Cinverse)  # Inverse of the inverse =  Cov
    Dstatinverse = np.diag(
        Cinverse
    )  # extract only the diagonal statistical uncertainties for SNe = Dstat (Dstat, see Scolnic et al. 2018)
    Dstat = np.diag(Cmat) ** 1 / 2  # Diagonal elements

    # SNe data
    SNdata = pd.read_csv("tables_and_data/AVGdeltamulcfrom1to1048ORD.txt", sep="\t")
    # Columns are muobs	zHD	zhel  mu-M  mu-M_err
    SNdata = SNdata.sort_values(by=["zHD"])
    zSNe = SNdata["zHD"].to_numpy()
    zhel = SNdata["zhel"].to_numpy()  # zhel is the heliocentric redshift
    muobs = SNdata["muobs"].to_numpy()  # The mu_observed of the SNe are already in Mpc

    return muobs, Cinverse


# Distance modulus for SN case
def distance_luminosity_SN(O_m, H0):

    SNdata = pd.read_csv("tables_and_data/AVGdeltamulcfrom1to1048ORD.txt", sep="\t")
    SNdata = SNdata.sort_values(by=["zHD"])
    zSNe = SNdata["zHD"].to_numpy()
    O_l = 1 - O_m
    coefQ = (ckm / H0) * (1 + zSNe)  # Units of this is Mpc

    d_par = np.array([])
    for i in zSNe:
        Integ = quad(integrand, 0, i, args=(O_m, O_l))
        d_par = np.append(d_par, Integ[0])

    d_mpc = np.array(coefQ * d_par)  # Distance luminosity # Unit of it is Mpc
    d_cm = d_mpc * conversionfrom_cm_to_Mpc  # As flux unit is [erg s^(-1) cm^(-2)], 1 cm = 1/ (3.0857 *10^24) Mpc

    return d_mpc, d_cm


# Creating table of Calibrated Quasars with SNe
def calibQSO():
    QSOdata = pd.read_csv("tables_and_data/Quasarstable.txt", sep="\t")
    QSOdata = QSOdata.sort_values(by=["z"], ignore_index=True)
    QSO_calibdata = QSOdata[QSOdata["z"] < 2.26]  # 2.26 is the maximum z in SN data
    QSO_calibdata.to_csv("tables_and_data/QSO_2065.txt", sep="\t")


# Flat cosmological model
def integrand(z, O_m, O_l):
    return 1 / (((1 + z) ** 3 * O_m + O_l) ** (1 / 2))


# Likelihoods

# Distance modulus likelihood for Quasars
if args.Equation == "dm":
    def my_like(g, b, sv, O_m, H0):

        (
            z,
            logFluxUV,
            logFluxUVerr,
            logFluxX,
            logFluxXerr,
            logLumUV,
            logLumX,
            logLumUVerr,
            logLumXerr,
        ) = QSO_data_read(
            data=data, O_m=O_m, H0=H0
        )  # Only fluxes are needed in DM
        d_mpc, d_cm = distance_luminosity_Q(O_m, H0)
        logdl_th = np.log10(d_mpc)
        muth = 5 * logdl_th + 25
        logdl_obs = -0.5 * (((g * logFluxUV + b - logFluxX) / (g - 1)) + np.log10(4 * np.pi))
        # As flux unit is [erg s^(-1) cm^(-2)], hence dl_obs unit is cm. But Theoretical value of Dl is in Mpc. Hence convert.
        logdl_obs = logdl_obs - np.log10(conversionfrom_cm_to_Mpc)  # As 1 cm = 1/ (3.0857 *10^24) Mpc
        muobs = 5 * logdl_obs + 25

        if args.Parameters in ["gbsv", "gbsvOm", "gbsvH0", "gbsvOmH0"]:  # In case of Without calibration
            sv_muobs_sq = (5**2) * (0.5**2) * (
                ((g / (g - 1)) ** 2) * logFluxUVerr**2 + ((1 / (g - 1)) ** 2) * logFluxXerr**2
            ) + (5 * sv) ** 2

        elif args.Parameters in ["Om", "H0", "OmH0"]:  # In case of calibration
            gerr = sigma0[0]
            berr = sigma0[1]
            sv_muobs_sq = (5**2) * (0.5**2) * (
                ((g / (g - 1)) ** 2) * logFluxUVerr**2
                + logLumUV**2 * (gerr / (g - 1) ** 2) ** 2
                + berr * 2
                + ((1 / (g - 1)) ** 2) * logFluxXerr**2
                + logLumX**2 * (gerr / (g - 1) ** 2) ** 2
            ) + (5 * sv) ** 2

        chi2 = np.sum((muth - muobs) ** 2 / (sv_muobs_sq))
        logliketotal = -(1 / 2) * np.sum(np.log(sv_muobs_sq)) - (1 / 2) * chi2

        if (
            args.Data == "both"
        ):  # will only g b sv feasible with both bec Om and H0 would be fixed, therefore likeSn would be fixed
            muobsSN, Cinverse = SN_data_read(O_m, H0)
            d_mpcSN, d_cmSN = distance_luminosity_SN(O_m, H0)
            logdl_thSN = np.log10(d_mpcSN)
            muthSN = 5 * logdl_thSN + 25
            Deltamu = muobsSN - muthSN

            chi2_SNe = np.matmul(Deltamu, np.matmul(Cinverse, Deltamu))
            loglikeSN = -(1 / 2) * chi2_SNe

            return logliketotal + loglikeSN
        else:
            return logliketotal


# Risaliti-Lusso relation likelihood
if args.Equation == "lum":

    def my_like(g, b, sv, O_m, H0):
        z, logFluxUV, logFluxUVerr, logFluxX, logFluxXerr, logLumUV, logLumX, logLumUVerr, logLumXerr = QSO_data_read(
            data=data, O_m=O_m, H0=H0
        )  # d is in Mpc

        logLumXth = g * logLumUV + b  # unit of Luminosity is erg /s
        if args.Parameters in ["gbsv", "gbsvOm", "gbsvH0", "gbsvOmH0"]:
            sigma2 = np.array(sv**2 + g**2 * logLumUVerr**2 + logLumXerr**2)

        elif args.Parameters in ["Om", "H0", "OmH0"]:
            gerr = sigma0[0]
            berr = sigma0[1]
            sigma2 = np.array(
                sv**2 + g**2 * logLumUVerr**2 + gerr**2 * logLumUV**2 + berr**2 + logLumXerr**2
            )

        chi2 = np.sum(np.array((logLumXth - logLumX) ** 2 / sigma2))
        return (-1 / 2) * np.sum(np.log(sigma2)) - 1 / 2 * chi2


# MCMC

# Uniform_onlyQ
def uniform_mcmc(likelihood):

    info = {"likelihood": {"agostini": likelihood}}

    if args.Calibration == "calib":

        if args.Parameters == "gbsv":
            guess = [0.665, 6.301, 0.230]
            g, b, sv = guess
            parameters = ["g", "b", "sv"]
            info["params"] = odict(
                [
                    ["g", {"prior": {"min": 0.0, "max": 0.8}, "ref": g, "proposal": 0.001}],
                    ["b", {"prior": {"min": 4.0, "max": 25.0}, "ref": b, "proposal": 0.001}],
                    ["sv", {"prior": {"min": 0.0, "max": 2.0}, "ref": sv, "proposal": 0.001}],
                    ["O_m", 0.3],
                    ["H0", 70.0],
                ]
            )

        if args.Parameters == "Om":
            O_m = 0.3
            parameters = ["O_m"]
            info["params"] = odict(
                [
                    ["g", mean0[0]],
                    ["b", mean0[1]],
                    ["sv", mean0[2]],
                    ["O_m", {"prior": {"min": 0, "max": 1}, "ref": O_m, "proposal": 0.001}],
                    ["H0", 70.0],
                ]
            )

        if args.Parameters == "H0":
            H0 = 70.0
            parameters = ["H0"]
            info["params"] = odict(
                [
                    ["g", mean0[0]],
                    ["b", mean0[1]],
                    ["sv", mean0[2]],
                    ["O_m", 0.3],
                    ["H0", {"prior": {"min": 50, "max": 100}, "ref": H0, "proposal": 0.001}],
                ]
            )

        if args.Parameters == "OmH0":
            O_m = 0.3
            H0 = 70.0
            parameters = ["O_m", "H0"]
            info["params"] = odict(
                [
                    ["g", mean0[0]],
                    ["b", mean0[1]],
                    ["sv", mean0[2]],
                    ["O_m", {"prior": {"min": 0, "max": 1}, "ref": O_m, "proposal": 0.001}],
                    ["H0", {"prior": {"min": 50, "max": 100}, "ref": H0, "proposal": 0.001}],
                ]
            )

    if args.Calibration == "nocalib":

        if args.Parameters == "gbsv":
            guess = [0.665, 6.301, 0.230]
            g, b, sv = guess
            parameters = ["g", "b", "sv"]
            info["params"] = odict(
                [
                    ["g", {"prior": {"min": 0.0, "max": 0.8}, "ref": g, "proposal": 0.001}],
                    ["b", {"prior": {"min": 4.0, "max": 25.0}, "ref": b, "proposal": 0.001}],
                    ["sv", {"prior": {"min": 0.0, "max": 2.0}, "ref": sv, "proposal": 0.001}],
                    ["O_m", 0.3],
                    ["H0", 70.0],
                ]
            )

        if args.Parameters == "gbsvOm":
            guess = [0.665, 6.301, 0.230, 0.3]
            g, b, sv, O_m = guess
            parameters = ["g", "b", "sv", "O_m"]
            info["params"] = odict(
                [
                    ["g", {"prior": {"min": 0.0, "max": 0.8}, "ref": g, "proposal": 0.001}],
                    ["b", {"prior": {"min": 4.0, "max": 25.0}, "ref": b, "proposal": 0.001}],
                    ["sv", {"prior": {"min": 0.0, "max": 2.0}, "ref": sv, "proposal": 0.001}],
                    ["O_m", {"prior": {"min": 0, "max": 1}, "ref": O_m, "proposal": 0.001}],
                    ["H0", 70.0],
                ]
            )

        if args.Parameters == "gbsvH0":
            guess = [0.665, 6.301, 0.230, 70.0]
            g, b, sv, H0 = guess
            parameters = ["g", "b", "sv", "H0"]
            info["params"] = odict(
                [
                    ["g", {"prior": {"min": 0.0, "max": 0.8}, "ref": g, "proposal": 0.001}],
                    ["b", {"prior": {"min": 4.0, "max": 25.0}, "ref": b, "proposal": 0.001}],
                    ["sv", {"prior": {"min": 0.0, "max": 2.0}, "ref": sv, "proposal": 0.001}],
                    ["O_m", 0.3],
                    ["H0", {"prior": {"min": 50, "max": 100}, "ref": H0, "proposal": 0.001}],
                ]
            )

        if args.Parameters == "gbsvOmH0":
            guess = [0.665, 6.301, 0.230, 0.3, 70.0]
            g, b, sv, O_m, H0 = guess
            parameters = ["g", "b", "sv", "O_m", "H0"]
            info["params"] = odict(
                [
                    ["g", {"prior": {"min": 0.0, "max": 0.8}, "ref": g, "proposal": 0.001}],
                    ["b", {"prior": {"min": 4.0, "max": 25.0}, "ref": b, "proposal": 0.001}],
                    ["sv", {"prior": {"min": 0.0, "max": 2.0}, "ref": sv, "proposal": 0.001}],
                    ["O_m", {"prior": {"min": 0, "max": 1}, "ref": O_m, "proposal": 0.001}],
                    ["H0", {"prior": {"min": 50, "max": 100}, "ref": H0, "proposal": 0.001}],
                ]
            )

    info["sampler"] = {
        "mcmc": {
            "burn_in": 300,
            "max_samples": 10000000,
            "Rminus1_stop": 0.1,
            "Rminus1_cl_stop": 0.2,
            "learn_proposal": True,
        }
    }

    updated_info, products = run(info)
    gdsamples = MCSamplesFromCobaya(updated_info, products.products()["sample"], ignore_rows=0.3)
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    # success = False
    # try:
    #     updated_info, products = run(info)
    #     success = True
    # except LoggedError as err:
    #     pass

    # # Did it work? (e.g. did not get stuck)
    # success = all(comm.allgather(success))

    # if not success and rank == 0:
    #     print("Sampling failed!")
    # all_chains = comm.gather(products.products()["sample"], root=0)

    # # Pass all of them to GetDist in rank = 0

    # if rank == 0:

    #     gdsamples = MCSamplesFromCobaya(updated_info, all_chains, ignore_rows=0.3)

    # # Manually concatenate them in rank = 0 for some custom manipulation,
    # # skipping 1st 3rd of each chain

    # copy_and_skip_1st_3rd = lambda chain: chain[int(len(chain) / 3) :]
    # if rank == 0:
    #     full_chain = copy_and_skip_1st_3rd(all_chains[0])
    #     for chain in all_chains[1:]:
    #         full_chain.append(copy_and_skip_1st_3rd(chain))
    #     # The combined chain is now `full_chain`

    gdplot = gdplt.getSubplotPlotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, parameters, filled=True)

    mean = gdsamples.getMeans()[: len(parameters)]
    sigma = np.sqrt(np.array(gdsamples.getVars()[: len(parameters)]))
    covmat = gdsamples.getCovMat().matrix[: len(parameters), : len(parameters)]

    return mean, sigma


if __name__ == "__main__":

    conversionfrom_cm_to_Mpc = 3.08567758 * 10**24  # 1 Mpc = 10^6 * 3.0857 * 10^16 * 10^2 cm
    ckm = 299792.458
    data = args.DataQuasar

    if args.Calibration == "calib":

        # MCMC for non-cosmological fit
        if args.Parameters == "gbsv":
            mean, sigma = uniform_mcmc(my_like)
            datafinal = np.vstack((mean, sigma))
            myheader = "Mean and 1 sigma of values"
            np.savetxt(
                f"{directory}{args.Calibration}/{args.Evolution}/{args.Evolution}_{args.Equation}_{args.Calibration}_{args.Data}_{args.Parameters}.txt",
                datafinal,
                fmt="%16.12e",
                header=myheader,
                delimiter=",",
            )

        elif args.Parameters in ["Om", "H0", "OmH0"]:
            # Extracting data from non cosmological MCMC run
            data_previous = np.loadtxt(
                f"{directory}{args.Calibration}/{args.Evolution}/{args.Evolution}_{args.Equation}_{args.Calibration}_{args.Data}_gbsv.txt",
                delimiter=",",
            )
            mean0 = data_previous[0, :]
            sigma0 = data_previous[1, :]

            # Running MCMC for cosmological fit
            mean1, sigma1 = uniform_mcmc(my_like)
            # Saving the mean and sigma
            datafinal = np.vstack((mean1, sigma1))
            myheader = "Mean and 1 sigma of values"
            np.savetxt(
                f"{directory}{args.Calibration}/{args.Evolution}/{args.Evolution}_{args.Equation}_{args.Calibration}_{args.Data}_{args.Parameters}.txt",
                datafinal,
                fmt="%16.12e",
                header=myheader,
                delimiter=",",
            )

    if args.Calibration == "nocalib":

        if args.Parameters in ["gbsv", "gbsvOm", "gbsvH0", "gbsvOmH0"]:
            mean, sigma = uniform_mcmc(my_like)
            datafinal = np.vstack((mean, sigma))
            myheader = "Mean and 1 sigma of values"
            np.savetxt(
                f"{directory}{args.Calibration}/{args.Evolution}/{args.Evolution}_{args.Equation}_{args.Calibration}_{args.Data}_{args.Parameters}.txt",
                datafinal,
                fmt="%16.12e",
                header=myheader,
                delimiter=",",
            )
    plt.savefig(
        f"{directory}{args.Calibration}/{args.Evolution}/{args.Evolution}_{args.Equation}_{args.Calibration}_{args.Data}_{args.Parameters}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
