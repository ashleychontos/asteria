# Stellar rotation

Estimate a stellar rotation period based on its location on an HR diagram. 

**Note:** If computation time is not a problem and you would like more physically-motivated estimates, please see my colleague Zach Claytor's software [kīahōkū](https://github.com/zclaytor/kiauhoku) that uses stellar isochrones x theoretical gyrochronology models.

### How it works
This program uses Gibbs Sampling, a special type of Markov Chain Monte Carlo (MCMC) sampling which is helpful for estimating properties from multidimensional parameter spaces for which direct sampling is not trivial. Assuming the rotation period strongly depends on the effective temperature and surface gravity of the star, we use the extensive observations available from the *Kepler* ([McQuillan, Mazeh, & Agrain 2014](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...24M/)) and K2 ([Reinhold & Hekker 2019](https://ui.adsabs.harvard.edu/abs/2020A%26A...635A..43R)) missions to statistically infer an approximate distribution. 

Essentially this is done by reducing this into a 1-D problem by "marginalizing" over the effective temperatures and surface gravities aka restricting the sample to stars with similar properties and measured rotation periods. Once we have the relevant sample of rotation periods, we construct the PDF, integrate to get the CDF and invert to get the quantile function. Therefore a random variable, X~[0,1], will successfully map back to the distribution to estimate a rotation period given the effective temperature and surface gravity of the star. 

P(prot | teff, logg)
