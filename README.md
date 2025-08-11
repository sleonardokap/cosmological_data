# ------ New Updates --------

[1] Added the Polychord sampler, which is way advanced than emcee sampler to obtain constraint on the higher dimensional paramter space.

[2] Added a Jupyter file for AIC and BIC calculation which will read the chains_new directory I have created. 

# ------ Older Updates --------

[1] Now I have updated the Planck likelihood. You can find these codes in BAO data directory. Use the planck code to fit your planck. It should be noted that this is not corresponding to the full Planck data. I have updated the code in which I have not yet given how to calculate the sound horizon distances at the recombination and photon baryon decoupling epoch. This advanced code will be released soon. 
[2] The code is fully integrated with the dynamical system technique. So, If you want to model any scalar field, it can be easily doable without changing the whole code. Just write the equations and obtain the H_sol and dl_sol. 
[3] So far I have not given any code for any other sampler except emcee. However, if you are using any complicated scalar field models, try to use dynesty or Polychord. They are advanced and efficient for obtaining higher dimensional parameter constraints. 
[4] In the stay tuned for more advanced algorithm. 

# --------- Other updates -----------

Now I have added the BAO data (SDSS+DESI R1 and R2) and corresponding error and covariane matrix (for DESI R2). I have not given the covariance matrix matrix for DESI R1 and SDSS measurements. 
I have shown how to calculate the several observables particularly: D_M, D_L, D_V and using that you can fit your cosmological model keeping r_d as a free parameter i.e., standard ruler at drag epoch (baryon photon decouple, z_d = 1059.90). 
The planck estimated the length to be near 147 KPc. This length must be greater than the r_s standard ruler at recombination epoch z_* = 1090. It is important to note that both these quantities are different and measured at different time scalae. 

# ------ Cite following papers -------------

If you are using my code or any part of this, kindly mention my github repo and cite my publications: 

[1]  @article{Hussain:2024qrd,
    author = "Hussain, Saddam and Nelleri, Sarath and Bhattacharya, Kaushik",
    title = "{Comprehensive study of k-essence model: dynamical system analysis and observational constraints from latest Type Ia supernova and BAO observations}",
    eprint = "2406.07179",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2025/03/025",
    journal = "JCAP",
    volume = "03",
    pages = "025",
    year = "2025"
}

[2]  @article{Hussain:2024jdt,
    author = "Hussain, Saddam",
    title = "{Particle Production Scenario in an Algebraically Coupled Quintessence Field with a Dark Matter Fluid}",
    eprint = "2403.10215",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2024",
    doi= "10.1016/j.cjph.2025.06.009",
    journal = "CJPh"
}

[3] @article{Hussain:2025vbo,
    author = "Hussain, Saddam and Arora, Simran and Rana, Yamuna and Rose, Benjamin and Wang, Anzhong",
    title = "{Interacting Scalar Fields as Dark Energy and Dark Matter in Einstein scalar Gauss Bonnet Gravity}",
    eprint = "2507.05207",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "7",
    year = "2025",
    doi =""
}

[4] @article{Myrzakulov:2025jpk,
    author = "Myrzakulov, Yerlan and Hussain, Saddam and Shahalam, M.",
    title = "{Phase space and data analyses of a nonminimally coupled scalar field system with a decaying dark energy model}",
    eprint = "2506.11755",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "6",
    year = "2025",
    doi =""
}

# ------ Additional Instruction -------

[1] For OHD data analysis with the LamdaCDM mdoel watch my tutorial here: https://youtu.be/0GDzoKncNO4 

[2] For the analysis of Supernova Pantheon+ data sets, watch my tutorial here: https://youtu.be/u88Goguzm9M

[3] I have updated the Planck likelihood. Check the folder BAO_DATA. 

[4] I have also updated the BAO DESI RII data with the covariance matrix. Now you can easily implement your models. 

[5] The code follows the dyanmical system framework which is a very powerful system to obtain stability of the system. So, now all the codes are highly optimized and efficient and uses a single framework which allows you to obtain all the constraint using a single framework. 

⚠️Note: While using the covariance matrix (after downloading from panthon+ website), you will probably get an error due to its shape. In order to resolve that you must follow the steps before reshaping. Open the file.cov in the vs code or any text editior, remove the first line written (1701) and save it. Now you can use the file for your calculation. 

[3] For BAO analysis see the video for complete understanding: https://youtu.be/iRYb6wQVaO8

If you are using any of the code: Kindly footnote the github link and cite the papers:

[a] @article{Hussain:2024jdt,
    author = "Hussain, Saddam",
    title = "{Non-adiabatic particle production scenario in algebraically coupled quintessence field with dark matter fluid}",
    eprint = "2403.10215",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2024"
}

[b] @article{Hussain:2024qrd,
    author = "Hussain, Saddam and Nelleri, Sarath and Bhattacharya, Kaushik",
    title = "{Comparative Analysis of $k$-essence and Quintessence Scalar Field Models: A Data Analysis Approach}",
    eprint = "2406.07179",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "6",
    year = "2024"
}

To know my work, check out my profile: https://inspirehep.net/authors/1852409


