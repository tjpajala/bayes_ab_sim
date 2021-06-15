import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
import matplotlib.pyplot as plt
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from typing import List, Union
from itertools import product
from tqdm import tqdm
import pandas as pd
from collections import Counter
from modules.utils import progress_print
from modules.faker import create_fake_ab_test_results, create_fake_likert_results
import seaborn as sns
from scipy import stats

def sim_single_experiment(exp_type:str, desired_certainty: float, true_results: dict, true_val: float,
                          min_n_obs: int = 10, inference_strategy: str = "analytical", rope_width: float = 0.01,
                          null_effect: float = 0.5, step: int = 50, print_progress: bool = False) -> pd.DataFrame:
    if exp_type == "binomial":
        val_field = "choice"
    elif exp_type == "likert":
        val_field="A"
    else:
        raise ValueError("Unknown exp_type, val_field is not defined for this experiment.")
    n_range = list(range(0, len(true_results[val_field]) + 1, step))
    n_range[0] = 1
    experiment_length = len(n_range)
    rope_area = (null_effect - rope_width, null_effect + rope_width)
    res = pd.DataFrame()
    res["hpdi_lower"] = [np.nan] * experiment_length
    res["hpdi_upper"] = [np.nan] * experiment_length
    res["n"] = [np.nan] * experiment_length
    res["percent_in_rope"] = [np.nan] * experiment_length
    res["conclusion"] = [np.nan] * experiment_length

    progress_print("Running {} experiment with true_val: {}, desired_certainty {}, ROPE {}\n".format(exp_type, true_val,
                                                                                                     desired_certainty,
                                                                                                     rope_area),
                   print_progress)

    # run inference for each step
    # we do not run sequentially so we can analyse the counterfactual situation of not breaking the exp
    if exp_type == "binomial":
        samples, hpdi = list(zip(*[run_binomial_inference(i, true_results, desired_certainty, inference_strategy) for i in n_range]))
    if exp_type == "likert":
        samples, hpdi = list(zip(*[run_likert_inference(i, true_results, desired_certainty, inference_strategy,
                                                        prior_mu=0,
                                                        prior_tau=5,
                                                        prior_a=0, prior_b=1) for i in n_range]))
    res["hpdi_lower"] = [x[0] for x in hpdi]
    res["hpdi_upper"] = [x[1] for x in hpdi]
    res["n"] = n_range
    res.iloc[0, res.columns.get_loc("n")] = 1
    res["desired_certainty"] = desired_certainty
    res["true_val"] = true_val
    res["ended"] = False
    res["stopping_reason"] = None
    res["conclusion"] = None

    for i in range(len(res)):
        nobs = res.loc[:, "n"][i]
        if ((nobs % 10 == 0) & (nobs >= min_n_obs)):
            hpdi = (res["hpdi_lower"][i], res["hpdi_upper"][i])
            percent_in_rope = max(0, min(rope_area[1], hpdi[1]) - max(rope_area[0], hpdi[0])) / (hpdi[1] - hpdi[0])
            res["percent_in_rope"] = percent_in_rope
            if exp_type == "binomial":
                cnt = dict(Counter(true_results[val_field][0:nobs]))
            elif exp_type == "likert":
                cnt = Counter((true_results["A"]-true_results["B"])[0:nobs] > 0)
                #print(cnt)
                cnt = {"A" if k == True else "B": v for k, v in cnt.items()}
            progress_print("With {} obs {}, {:.0%} HPDI is {:.3f} - {:.3f} ({:.3%} in ROPE)".format(nobs, cnt,
                                                                                                    desired_certainty,
                                                                                                    hpdi[0],
                                                                                                    hpdi[1],
                                                                                                    percent_in_rope),
                           print_progress)
            rope = check_rope(percent_in_rope)
            if rope == "uncertain":
                progress_print("Too much uncertainty at {} obs, should continue study.".format(nobs),
                               print_progress)
                pass
            elif rope == "no_diff":
                progress_print(
                    "HDI inside ROPE, so no practical difference at {} certainty!".format(desired_certainty),
                    print_progress)
                res.loc[res["n"] >= nobs, "ended"] = True
                res.loc[res["n"] >= nobs, "stopping_reason"] = "bayes"
                res.loc[res["n"] == nobs, "conclusion"] = "no_diff"
                progress_print("Breaking at {} obs.\n".format(nobs), print_progress)
                break
            elif rope == "signif_diff":
                progress_print("HDI outside ROPE, so signif. diff at {} certainty!".format(desired_certainty),
                               print_progress)
                res.loc[res["n"] >= nobs, "ended"] = True
                res.loc[res["n"] >= nobs, "stopping_reason"] = "bayes"
                if hpdi[0] > null_effect:
                    res.loc[res["n"] >= nobs, "conclusion"] = "A"
                    progress_print("Conclude: A is better!", print_progress)
                elif hpdi[1] < null_effect:
                    res.loc[res["n"] >= nobs, "conclusion"] = "B"
                    progress_print("Conclude: B is better!", print_progress)
                else:
                    # print("Something is wrong, HDI {} outside ROPE {} but unsure".format(hpdi, rope_area))
                    # res.loc[res["n"] >= nobs,"conclusion"] = "WHAT"
                    raise ValueError(
                        "Something is wrong, HDI {} outside ROPE {} but unsure".format(hpdi, rope_area))
                progress_print("Breaking at {} obs.\n".format(nobs), print_progress)
                break

    # if experiment didn't break prematurely, change last obs to "ended"
    if sum(res["ended"]) == 0:
        res.iloc[-1, res.columns.get_loc("ended")] = True
        res.iloc[-1, res.columns.get_loc("stopping_reason")] = "end_of_experiment"
        progress_print("Ending experiment at {} obs because reached end of data.".format(nobs), print_progress)

    res["correct"] = res.apply(lambda row: row["conclusion"] == "A", axis=1)
    return res


def check_rope(percent_in_rope):
    if percent_in_rope > 0.99:
        return "no_diff"
    elif percent_in_rope < 0.01:
        return "signif_diff"
    else:
        return "uncertain"


def run_binomial_inference(i, true_results, desired_certainty, inference_strategy, prior_w=10, prior_l=10):
    W = sum(true_results["choice"][0:i] == 0)
    L = sum(true_results["choice"][0:i] == 1)
    if inference_strategy == "svi":
        def model(W, L, prior_w, prior_l):
            prior = numpyro.distributions.Beta(prior_w, prior_l)
            p = numpyro.sample("p", prior)  # prior
            numpyro.sample("W", dist.Binomial(W + L, p), obs=W)  # binomial likelihood

        guide = AutoLaplaceApproximation(model)
        svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(), W=W, L=L, prior=dist.Beta(1, 1))
        params, losses = svi.run(random.PRNGKey(0), 1000, progress_bar=False)
        # display summary of quadratic approximation
        post_samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
    if inference_strategy == "analytical":
        post_samples = dict()
        post_samples["p"] = dist.Beta(W + prior_w, L + prior_l).sample(random.PRNGKey(1), (1000,))
    hpdi = numpyro.diagnostics.hpdi(post_samples["p"], prob=desired_certainty)
    return post_samples, hpdi


def run_likert_inference(i, true_results, desired_certainty, inference_strategy,
                         prior_mu: Union[numpyro.distributions.Distribution, float],
                         prior_tau: Union[numpyro.distributions.Distribution, float],
                         prior_a: float, prior_b: float):
    val_diff = true_results["A"][0:i] - true_results["B"][0:i]
    if inference_strategy == "svi":
        def model(val_diff, mu_prior, tau_prior):
            # print(val_diff)
            mu = numpyro.sample("diff", mu_prior)
            sd = numpyro.sample("sd", tau_prior)
            numpyro.sample("obs", dist.Normal(mu, sd), obs=val_diff)  # Normal likelihood

        guide = AutoLaplaceApproximation(model)
        svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(), val_diff=val_diff, mu_prior=prior_mu, tau_prior=prior_tau)
        params, losses = svi.run(random.PRNGKey(0), 1000, progress_bar=False)
        # display summary of quadratic approximation
        post_samples = guide.sample_posterior(random.PRNGKey(1), params, (10000,))
        #numpyro.diagnostics.print_summary(post_samples, prob=desired_certainty, group_by_chain=False)
    if inference_strategy == "analytical":
        post_samples = dict()
        b = prior_b
        v = prior_tau
        u0 = prior_mu
        a = prior_a
        for i in range(len(val_diff)):
            x = val_diff[i]
            a = a + 0.5
            b = b + 0.5 * v / (v + 1) * (x - u0) ** 2
            u0 = (v * u0 + x) / (v + 1)
            v += 1
        std = ((2 * b * (v + 1) / (v * v)) ** 0.5)
        t_star = stats.t.ppf(1-(1-desired_certainty)/2, v - 1)
        add = t_star * (b / (a * v)) ** 0.5
        hpdi = sorted((float(u0 - add), float(u0 + add)))
        #print("Final updated u0: {:.2f}, sd: {:.2f}, v: {}, var: {:.2f}".format(u0, std, v, std))
        #print("Computed posterior vals for diff: mean {:.2f}, hpdi {}".format(u0, hpdi))
        post_samples["p"] = None
    return post_samples, hpdi


def run_single_binomial_exp(prob, cert, experiment_no, n_resp, print_graphs, print_progress, null_effect):
    exp_no = "_".join([str(prob), str(cert), str(experiment_no + 1)])
    progress_print("Creating fake data for experiment {}, true_prob {}, desired cert {}, n_obs {}".format(experiment_no,
                                                                                                          prob, cert,
                                                                                                          n_resp),
                   print_progress)
    sim_data = create_fake_ab_test_results(n=n_resp, prob_a=prob)
    progress_print("Simulated data has {} obs".format(Counter(sim_data["choice"])), print_progress)
    single_res = sim_single_experiment(exp_type="binomial", desired_certainty=cert, true_results=sim_data,
                                       true_val=prob, print_progress=print_progress, null_effect=null_effect)
    single_res["exp_no"] = exp_no
    cutoff = (single_res.ended == True).idxmax()
    cutoff_n = single_res.loc[(single_res.ended == True).idxmax(), "n"]

    if print_graphs:
        d = pd.melt(single_res, id_vars=["n"], value_vars=["hpdi_lower", "hpdi_upper", "true_val"])
        fig = sns.lineplot(data=d, x="n", y="value", hue="variable")
        plt.axhline(0.5, linestyle='--', color="blue")
        plt.axvline(cutoff_n, linestyle="--", color="red")
        plt.show()
    return single_res


def run_single_likert_exp(prob, cert, experiment_no, n_resp, print_graphs, print_progress, null_effect):
    exp_no = "_".join([str(prob), str(cert), str(experiment_no + 1)])
    progress_print("Creating fake data for experiment {}, true_prob {}, desired cert {}, n_obs {}".format(experiment_no,
                                                                                                          prob, cert,
                                                                                                          n_resp),
                   print_progress)
    sim_data = create_fake_likert_results(n=n_resp, center=0., std=1., diff=prob, round_responses=False)
    diffs = sim_data["A"] > sim_data["B"]
    cnt = Counter(diffs)
    cnt = {"A" if k == True else "B": v for k, v in cnt.items()}
    progress_print("Simulated data has {} obs".format(cnt), print_progress)
    single_res = sim_single_experiment(exp_type="likert", desired_certainty=cert, true_results=sim_data,
                                       true_val=prob, print_progress=print_progress, null_effect=null_effect,
                                       inference_strategy="analytical")
    single_res["exp_no"] = exp_no
    cutoff = (single_res.ended == True).idxmax()
    cutoff_n = single_res.loc[(single_res.ended == True).idxmax(), "n"]

    if print_graphs:
        d = pd.melt(single_res, id_vars=["n"], value_vars=["hpdi_lower", "hpdi_upper", "true_val"])
        fig = sns.lineplot(data=d, x="n", y="value", hue="variable")
        plt.axhline(0.5, linestyle='--', color="blue")
        plt.axvline(cutoff_n, linestyle="--", color="red")
        plt.show()
    return single_res


def sim_many_experiments(exp_type: str, n_exp: int, true_vals: List[float],
                         desired_cert_list: List[float], n_resp: int = 500,
                         print_graphs=False, print_progress=False, null_effect=0.5) -> pd.DataFrame:
    params = list(product(true_vals, desired_cert_list, range(1, n_exp + 1)))
    progress_print(params, print_progress)
    all_res = [None] * len(params)
    progress_print("Running total of {} experiments.".format(len(all_res)), print_progress)
    i = 1
    if exp_type == "binomial":
        all_res = [run_single_binomial_exp(prob, cert, n, n_resp, print_graphs, print_progress, null_effect=null_effect) for prob, cert, n in tqdm(params)]
    elif exp_type == "likert":
        all_res = [run_single_likert_exp(prob, cert, n, n_resp, print_graphs, print_progress, null_effect=null_effect) for prob, cert, n in
                   tqdm(params)]
    return pd.concat(all_res)

