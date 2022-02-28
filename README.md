# Stellar rotation

Estimate a stellar rotation period based on its location on an HR diagram. 

**Note:** If computation time is not a problem and you would like more physically-motivated estimates, please see my colleague Zach Claytor's software [kīahōkū](https://github.com/zclaytor/kiauhoku) that uses stellar isochrones x theoretical gyrochronology models.

### How it works
This program uses a Gibbs Sampler, which is a special type of Markov Chain Monte Carlo (MCMC) sampling algorithm helpful for estimating properties from joint parameter spaces for which direct sampling is not trivial. Specifically in this case, we assume the rotation period depends on the effective temperature and surface gravity of the star, the two variables which are not independent. Therefore, we statistically infer the approximate joint distribution using observations of rotation periods from the well-known *Kepler* ([McQuillan, Mazeh, & Agrain 2014](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...24M/)) and K2 ([Reinhold & Hekker 2019](https://ui.adsabs.harvard.edu/abs/2020A%26A...635A..43R)). Next we integrate the distribution to get the CDF and then take the inverse (referred to as the quantile function) in order to sample directly from the approximated distribution. 

P(prot | teff, logg)
