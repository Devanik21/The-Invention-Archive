#!/usr/bin/env python3
"""Deterministic synthetic pipeline test for Silica Light Meters.

This is not RCWA/FDTD and does not use empirical species, light-preference, or
sediment-core data. Every synthetic quantity is labeled and archived.
"""
from __future__ import annotations
import csv, json, platform, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

SEED = 20261018
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
TITLE = "Silica Light Meters"

# Coherent, restrained figure palette; no color is used as the only encoding.
INK = "#19324A"
TEAL = "#197C83"
BLUE = "#456990"
GOLD = "#C08A38"
CORAL = "#D36B54"
SLATE = "#64748B"
PALE = "#E9F1F2"


def setup_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "figure.dpi": 130,
        "savefig.dpi": 240, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#94A3B8",
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": SLATE, "ytick.color": SLATE,
        "grid.color": "#D8E2E5", "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "lines.linewidth": 2.0,
    })


def savefig(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", metadata={"Title": stem.replace("_", " ")})
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v) for k,v in row.items()})


def surrogate_transmission(wavelength_nm, pitch, pore_diameter, porosity, layer_index):
    """Hypothetical normalized spectral surrogate; deliberately not a Maxwell solver."""
    wl = np.asarray(wavelength_nm, dtype=float)
    center1 = 425.0 + 210.0 * pitch + 35.0 * (layer_index - 0.5)
    center2 = 650.0 - 115.0 * porosity + 22.0 * (pore_diameter - 0.4)
    width1 = 24.0 + 26.0 * pore_diameter
    width2 = 32.0 + 18.0 * porosity
    val = 0.18 + 0.42 * np.exp(-0.5*((wl-center1)/width1)**2) + 0.24*np.exp(-0.5*((wl-center2)/width2)**2)
    val *= (0.88 + 0.16 * np.clip(pore_diameter / max(pitch, 0.05), 0.05, 0.95))
    return np.clip(val, 0.0, 1.0)


def surface_spectrum(wl, centre=555.0, width=105.0):
    return 0.45 + np.exp(-0.5*((wl-centre)/width)**2)


def attenuation_coeff(wl, water_type):
    # Arbitrary synthetic coefficients, in inverse metres, for pedagogical illustration only.
    base, cdom, particles = {"clear": (0.08, 0.06, 0.02), "mixed": (0.12, 0.23, 0.09), "turbid": (0.20, 0.48, 0.22)}[water_type]
    return base + cdom*np.exp(-(wl-400)/82.0) + particles*(550.0/wl)**1.1


def r2(y, pred):
    den = float(np.sum((y-np.mean(y))**2))
    return 1.0-float(np.sum((y-pred)**2))/den if den > 1e-12 else 0.0


def ols_predict(x_train, y_train, x_test):
    # Numerically stable least squares with an explicit intercept column.
    xt = np.column_stack([np.ones(len(x_train)), x_train])
    xe = np.column_stack([np.ones(len(x_test)), x_test])
    coef, *_ = np.linalg.lstsq(xt, y_train, rcond=None)
    return xe @ coef


def group_fixed_design(clade, train_idx, test_idx):
    # The clade intercept is a coarse synthetic control, not a fitted phylogenetic tree.
    groups = np.unique(clade)
    d = np.column_stack([(clade == g).astype(float) for g in groups[1:]])
    return d[train_idx], d[test_idx]


def simulate_species(rng, n_species=144, n_clades=8):
    clade = np.repeat(np.arange(n_clades), n_species//n_clades)
    if len(clade) < n_species:
        clade = np.concatenate([clade, np.arange(n_species-len(clade)) % n_clades])
    rng.shuffle(clade)
    clade_effect = rng.normal(0, 0.65, n_clades)
    # Partial clade structure in geometry intentionally creates a confounding opportunity.
    pitch = np.clip(rng.beta(2.0, 2.6, n_species) + 0.08*(clade-(n_clades-1)/2)/(n_clades-1), 0.08, 0.95)
    diameter = np.clip(0.10 + 0.74*rng.beta(2.1, 2.4, n_species) + 0.05*(clade-(n_clades-1)/2)/(n_clades-1), 0.05, 0.92)
    porosity = np.clip(0.15 + 0.70*rng.beta(2.3, 2.2, n_species), 0.05, 0.95)
    layer = np.clip(rng.uniform(0.1, 1.0, n_species) + 0.08*(clade-(n_clades-1)/2)/(n_clades-1), 0.05, 1.1)
    wl = np.linspace(400, 700, 61)
    action = 0.55*np.exp(-0.5*((wl-445)/28)**2) + 0.60*np.exp(-0.5*((wl-675)/23)**2) + 0.1
    light = surface_spectrum(wl, 545, 120)
    T = np.vstack([surrogate_transmission(wl, pitch[i], diameter[i], porosity[i], layer[i]) for i in range(n_species)])
    raw_q = np.trapezoid(T*light[None,:]*action[None,:], wl, axis=1)
    optical = (raw_q - raw_q.mean()) / (raw_q.std() + 1e-10)
    raw_pore = (diameter - diameter.mean())/(diameter.std()+1e-10)
    clade_z = (clade_effect[clade] - np.mean(clade_effect[clade]))/(np.std(clade_effect[clade])+1e-10)
    df = []
    for i in range(n_species):
        df.append({"synthetic_taxon":f"sp_{i+1:03d}", "synthetic_clade":int(clade[i]), "pitch_norm":pitch[i], "pore_diameter_norm":diameter[i], "porosity_norm":porosity[i], "layer_index_norm":layer[i], "surrogate_optical_score":optical[i], "raw_pore_score":raw_pore[i], "clade_effect_latent":clade_z[i]})
    return {"clade":clade,"clade_z":clade_z,"pitch":pitch,"diameter":diameter,"porosity":porosity,"layer":layer,"optical":optical,"raw_pore":raw_pore,"wl":wl,"T":T,"species_rows":df,"n_clades":n_clades}


def cv_scenarios(rng, species, n_reps=90):
    clade = species["clade"]
    optical = species["optical"]
    pore = species["raw_pore"]
    n = len(clade)
    out = []
    for world in ["optical_signal", "phylogeny_confounded", "no_direct_signal"]:
        for rep in range(n_reps):
            # New latent clade effects/noise per replicate; synthetic worlds are fully declared.
            latent_clade = rng.normal(0, 0.65, species["n_clades"])[clade]
            if world == "optical_signal":
                y = 0.80*optical + 0.25*latent_clade + rng.normal(0, 0.75, n)
            elif world == "phylogeny_confounded":
                y = 0.85*latent_clade + rng.normal(0, 0.75, n)
            else:
                y = rng.normal(0, 1.0, n)
            y = (y-y.mean())/(y.std()+1e-10)
            idx = np.arange(n); rng.shuffle(idx)
            test_idx = idx[:int(0.28*n)]; train_idx = idx[int(0.28*n):]
            # Keep clades represented in train/test by constructing a within-clade split.
            train_l=[]; test_l=[]
            for g in np.unique(clade):
                members = np.where(clade==g)[0].copy(); rng.shuffle(members)
                ntest = max(1, int(0.28*len(members)))
                test_l.extend(members[:ntest]); train_l.extend(members[ntest:])
            train_idx=np.asarray(train_l); test_idx=np.asarray(test_l)
            dtr, dte = group_fixed_design(clade, train_idx, test_idx)
            Xp_tr=np.column_stack([dtr,pore[train_idx]])
            Xp_te=np.column_stack([dte,pore[test_idx]])
            Xo_tr=np.column_stack([dtr,optical[train_idx]])
            Xo_te=np.column_stack([dte,optical[test_idx]])
            pred_p=ols_predict(Xp_tr,y[train_idx],Xp_te)
            pred_o=ols_predict(Xo_tr,y[train_idx],Xo_te)
            out.append({"world":world,"replicate":rep,"r2_raw_pore":r2(y[test_idx],pred_p),"r2_spectral_surrogate":r2(y[test_idx],pred_o),"delta_r2_spectral_minus_pore":r2(y[test_idx],pred_o)-r2(y[test_idx],pred_p),"n_train":len(train_idx),"n_test":len(test_idx),"n_clades":int(len(np.unique(clade)))})
    return out


def make_figures(species, cv_rows):
    setup_style(); FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    wl=species["wl"]
    # Plot 1: surrogate spectra
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    ids=[7,34,68,101,132]
    for j,i in enumerate(ids):
        ax.plot(wl, species["T"][i], label=f"synthetic taxon {i+1}", color=[TEAL,BLUE,GOLD,CORAL,SLATE][j])
    ax.set(xlabel="Wavelength (nm)",ylabel="Normalized surrogate response",title="Geometry-to-spectrum feature generator")
    ax.set_ylim(0,1); ax.grid(True,alpha=.65); ax.legend(ncol=2,frameon=False)
    ax.text(.01,-.28,"Synthetic Gaussian-resonance surrogate; not RCWA/FDTD or measured data.", transform=ax.transAxes, fontsize=7.5,color=SLATE)
    savefig(fig,"fig01_surrogate_spectra")
    spectra_rows=[]
    for i in ids:
        for w,t in zip(wl,species["T"][i]): spectra_rows.append({"synthetic_taxon":f"sp_{i+1:03d}","wavelength_nm":w,"surrogate_transmission":t})
    write_csv(RES/"surrogate_spectra.csv",["synthetic_taxon","wavelength_nm","surrogate_transmission"],spectra_rows)
    # Plot 2: water-column attenuation
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    for typ,col in [("clear",TEAL),("mixed",BLUE),("turbid",CORAL)]:
        for depth,ls in [(2,"-"),(8,"--")]:
            e=surface_spectrum(wl,545,120)*np.exp(-attenuation_coeff(wl,typ)*depth)
            e=e/e.max()
            ax.plot(wl,e,color=col,linestyle=ls,label=f"{typ}, depth {depth} m")
    ax.set(xlabel="Wavelength (nm)",ylabel="Relative downwelling irradiance",title="Water-column attenuation is a spectral nuisance channel")
    ax.grid(True,alpha=.65); ax.legend(ncol=2,frameon=False)
    ax.text(.01,-.28,"Synthetic coefficients are arbitrary; relative curves only.",transform=ax.transAxes,fontsize=7.5,color=SLATE)
    savefig(fig,"fig02_attenuation")
    att_rows=[]
    for typ in ["clear","mixed","turbid"]:
      for depth in [0,2,5,8,12]:
       for w in wl:
        att_rows.append({"water_type_synthetic":typ,"depth_m_synthetic":depth,"wavelength_nm":w,"attenuation_m_inv_synthetic":attenuation_coeff(np.array([w]),typ)[0],"relative_irradiance_synthetic":float(surface_spectrum(np.array([w]),545,120)[0]*np.exp(-attenuation_coeff(np.array([w]),typ)[0]*depth))})
    write_csv(RES/"attenuation_curves.csv",["water_type_synthetic","depth_m_synthetic","wavelength_nm","attenuation_m_inv_synthetic","relative_irradiance_synthetic"],att_rows)
    # Plot 3: incremental skill across synthetic worlds
    by={}
    for world in ["optical_signal","phylogeny_confounded","no_direct_signal"]:
        by[world]=[r["delta_r2_spectral_minus_pore"] for r in cv_rows if r["world"]==world]
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    vals=[by[k] for k in by]
    parts=ax.violinplot(vals,showmeans=True,showmedians=True,showextrema=False)
    for body,col in zip(parts['bodies'],[TEAL,GOLD,SLATE]): body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(.55)
    parts['cmeans'].set_color(INK); parts['cmedians'].set_color(CORAL)
    ax.axhline(0,color=INK,lw=1,ls="--")
    ax.set_xticks([1,2,3], ["Optical signal","Phylogeny confound","No direct signal"])
    ax.set(ylabel=r"Held-out $R^2$ difference (spectral - pore)",title="Synthetic negative controls test whether the pipeline can fail")
    ax.grid(axis="y",alpha=.55)
    ax.text(.01,-.25,"90 simulation replicates per world; coarse clade intercept; not real predictive performance.",transform=ax.transAxes,fontsize=7.5,color=SLATE)
    savefig(fig,"fig03_incremental_prediction")
    # Plot 4: a scalar-observation inverse ambiguity surface
    mu=np.linspace(450,650,91); atten=np.linspace(.05,.65,91)
    wl2=np.linspace(400,700,301); g=species
    idx=68; trans=species["T"][idx]
    action=0.55*np.exp(-0.5*((wl2-445)/28)**2)+0.60*np.exp(-0.5*((wl2-675)/23)**2)+.1
    obs=np.zeros((len(atten),len(mu)))
    for i,a in enumerate(atten):
      for j,m in enumerate(mu):
        e=np.exp(-0.5*((wl2-m)/62)**2)
        obs[i,j]=np.trapezoid(e*np.exp(-(0.05+a*np.exp(-(wl2-400)/180))*5)*np.interp(wl2,wl,trans)*action,wl2)
    target=np.quantile(obs,.53)
    err=np.abs(obs-target)/(np.max(obs)-np.min(obs)+1e-12)
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    cf=ax.contourf(mu,atten,err,levels=np.linspace(0,0.5,15),cmap="viridis_r")
    cs=ax.contour(mu,atten,obs,levels=7,colors="white",linewidths=.7,alpha=.9)
    ax.clabel(cs,inline=True,fontsize=7,fmt="%.2f")
    fig.colorbar(cf,ax=ax,label="Normalized absolute mismatch")
    ax.set(xlabel="Synthetic incident-spectrum centre (nm)",ylabel="Synthetic attenuation scale",title="Different target/nuisance pairs can match one observation")
    ax.text(.01,-.25,"One scalar summary, two varying latent quantities; illustrative model only.",transform=ax.transAxes,fontsize=7.5,color=SLATE)
    savefig(fig,"fig04_inverse_nonidentifiability")
    inv_rows=[]
    for i,a in enumerate(atten):
      for j,m in enumerate(mu): inv_rows.append({"incident_centre_nm_synthetic":m,"attenuation_scale_synthetic":a,"scalar_observation_synthetic":obs[i,j],"normalized_mismatch_to_target_synthetic":err[i,j]})
    write_csv(RES/"inverse_identifiability_grid.csv",["incident_centre_nm_synthetic","attenuation_scale_synthetic","scalar_observation_synthetic","normalized_mismatch_to_target_synthetic"],inv_rows)
    # Graph 5: causal dependency graph
    G=nx.DiGraph()
    edges=[("Ancestry","Geometry"),("Light history","Ecological selection"),("Nutrients / temperature","Ecological selection"),("Ecological selection","Geometry"),("Geometry","Optical response"),("Surface spectrum","Underwater irradiance"),("Water attenuation","Underwater irradiance"),("Optical response","Measured exposure"),("Underwater irradiance","Measured exposure"),("Growth / preference","Ecological selection"),("Measured exposure","Growth / preference"),("Preservation","Core morphology"),("Geometry","Core morphology"),("Core morphology","Inverse posterior"),("Independent core proxy","Validation gate"),("Inverse posterior","Validation gate")]
    G.add_edges_from(edges)
    pos={"Ancestry":(0,3),"Light history":(1.7,4),"Nutrients / temperature":(1.7,2.9),"Ecological selection":(3.6,3.55),"Geometry":(5.5,3.55),"Optical response":(7.4,4.3),"Surface spectrum":(0.0,1.0),"Water attenuation":(1.7,.2),"Underwater irradiance":(3.7,.65),"Measured exposure":(5.8,1.2),"Growth / preference":(7.5,2.5),"Preservation":(5.4,-.6),"Core morphology":(7.4,-.4),"Inverse posterior":(9.3,.3),"Independent core proxy":(9.2,-0.95),"Validation gate":(11.0,-.3)}
    fig,ax=plt.subplots(figsize=(10.5,4.15))
    labels={"Ancestry":"Ancestry","Light history":"Light\nhistory","Nutrients / temperature":"Nutrients /\ntemperature","Ecological selection":"Ecological\nselection","Geometry":"Geometry","Optical response":"Optical\nresponse","Surface spectrum":"Surface\nspectrum","Water attenuation":"Water\nattenuation","Underwater irradiance":"Underwater\nirradiance","Measured exposure":"Measured\nexposure","Growth / preference":"Growth /\npreference","Preservation":"Preservation","Core morphology":"Core\nmorphology","Inverse posterior":"Inverse\nposterior","Independent core proxy":"Independent\ncore proxy","Validation gate":"Validation\ngate"}
    nx.draw_networkx_nodes(G,pos,node_size=2850,node_shape="s",node_color=[PALE if n not in ["Light history","Inverse posterior","Validation gate"] else "#F5E8CE" for n in G.nodes],edgecolors=TEAL,linewidths=1.1,ax=ax)
    nx.draw_networkx_labels(G,pos,labels=labels,font_size=6.5,font_color=INK,ax=ax)
    nx.draw_networkx_edges(G,pos,arrows=True,arrowstyle="-|>",arrowsize=10,width=0.9,edge_color=SLATE,connectionstyle="arc3,rad=0.02",min_source_margin=22,min_target_margin=25,ax=ax)
    ax.set_title("Causal dependencies: optical forward physics is only one layer",loc="left",color=INK,pad=10)
    ax.set_xlim(-.8,12); ax.set_ylim(-1.9,4.8); ax.axis("off")
    savefig(fig,"fig05_causal_graph")
    # Graph 6: dependency graph of modeling features
    G2=nx.DiGraph()
    G2.add_edges_from([("SEM geometry","RCWA / FDTD response"),("Optical constants","RCWA / FDTD response"),("Orientation / polarization","RCWA / FDTD response"),("RCWA / FDTD response","Spectral features"),("Raw pore metrics","Morphology baseline"),("Spectral features","Physics-informed model"),("Phylogeny","Hierarchical control"),("Habitat covariates","Hierarchical control"),("Hierarchical control","Species preference posterior"),("Physics-informed model","Species preference posterior"),("Species preference posterior","Core inverse model"),("Attenuation state","Core inverse model"),("Chronology / preservation","Core inverse model"),("Core inverse model","Posterior + identifiability gate")])
    pos2={"SEM geometry":(0,3),"Optical constants":(0,2),"Orientation / polarization":(0,1),"RCWA / FDTD response":(2,2),"Spectral features":(4,2.6),"Raw pore metrics":(2,.2),"Morphology baseline":(4,.2),"Physics-informed model":(6,2.4),"Phylogeny":(4,-.8),"Habitat covariates":(4,-1.7),"Hierarchical control":(6,-.9),"Species preference posterior":(8,1.0),"Attenuation state":(8,-.2),"Chronology / preservation":(8,-1.2),"Core inverse model":(10,.0),"Posterior + identifiability gate":(12,.0)}
    fig,ax=plt.subplots(figsize=(10.8,4.15))
    labels2={"SEM geometry":"SEM\ngeometry","Optical constants":"Optical\nconstants","Orientation / polarization":"Orientation /\npolarization","RCWA / FDTD response":"RCWA / FDTD\nresponse","Spectral features":"Spectral\nfeatures","Raw pore metrics":"Raw pore\nmetrics","Morphology baseline":"Morphology\nbaseline","Physics-informed model":"Physics-\ninformed model","Phylogeny":"Phylogeny","Habitat covariates":"Habitat\ncovariates","Hierarchical control":"Hierarchical\ncontrol","Species preference posterior":"Preference\nposterior","Attenuation state":"Attenuation\nstate","Chronology / preservation":"Chronology /\npreservation","Core inverse model":"Core inverse\nmodel","Posterior + identifiability gate":"Posterior +\nID gate"}
    nx.draw_networkx_nodes(G2,pos2,node_size=2850,node_shape="s",node_color=PALE,edgecolors=TEAL,linewidths=1.0,ax=ax)
    nx.draw_networkx_labels(G2,pos2,labels=labels2,font_size=6.3,font_color=INK,ax=ax)
    nx.draw_networkx_edges(G2,pos2,arrows=True,arrowstyle="-|>",arrowsize=9,width=.85,edge_color=SLATE,connectionstyle="arc3,rad=.02",min_source_margin=22,min_target_margin=25,ax=ax)
    ax.set_title("Proposed dependencies between measurements, models and posterior",loc="left",color=INK,pad=10)
    ax.set_xlim(-.8,13.0); ax.set_ylim(-2.5,3.8); ax.axis("off")
    savefig(fig,"fig06_feature_graph")
    # Flow diagram 7: concise labels are used because axis-level text wrapping
    # can vary between interactive, PNG, and PDF renderers.
    flow1=[("1. Geometry","SEM/TEM; error"),("2. Optics","Solver vs spectra"),("3. Preference","Controlled gradient"),("4. Model test","M0-M2; held-out taxa"),("5. Transfer","Held-out site/region"),("6. Core gate","After prior gates")]
    fig,ax=plt.subplots(figsize=(8.5,2.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3)
    xs=np.linspace(.9,9.1,len(flow1))
    for i,(head,sub) in enumerate(flow1):
        box=plt.Rectangle((xs[i]-.72,1.0),1.44,1.10,facecolor=PALE,edgecolor=TEAL,linewidth=1.1,joinstyle="round")
        ax.add_patch(box); ax.text(xs[i],1.72,head,ha="center",va="center",fontsize=7.0,fontweight="bold",color=INK)
        ax.text(xs[i],1.28,sub,ha="center",va="center",fontsize=5.8,color=SLATE,wrap=True)
        if i<len(flow1)-1: ax.annotate("",(xs[i+1]-.76,1.55),(xs[i]+.76,1.55),arrowprops={"arrowstyle":"->","color":GOLD,"lw":1.4})
    ax.text(5,2.65,"Conditional validation sequence: every stage has a stop rule",ha="center",fontsize=10,color=INK)
    ax.text(5,.45,"A failed upstream gate blocks downstream paleoclimate claims.",ha="center",fontsize=8,color=CORAL)
    savefig(fig,"fig07_validation_flow")
    # Flow diagram 8: inference gate with short labels sized to fit each node.
    flow2=[("Core traits","Morphology + error"),("Posterior","Light + nuisance"),("ID gate","Rank / sensitivity"),("External test","Proxy / modern history")]
    fig,ax=plt.subplots(figsize=(8.5,2.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3)
    xs=[1.2,3.75,6.25,8.8]
    for i,(head,sub) in enumerate(flow2):
        box=plt.Rectangle((xs[i]-.88,1.05),1.76,1.05,facecolor=("#F5E8CE" if i==2 else PALE),edgecolor=(GOLD if i==2 else TEAL),linewidth=1.2)
        ax.add_patch(box); ax.text(xs[i],1.72,head,ha="center",va="center",fontsize=7.1,fontweight="bold",color=INK)
        ax.text(xs[i],1.28,sub,ha="center",va="center",fontsize=5.9,color=SLATE)
        if i<len(xs)-1: ax.annotate("",xy=(xs[i+1]-.9,1.58),xytext=(xs[i]+.9,1.58),arrowprops={"arrowstyle":"->","color":GOLD,"lw":1.3})
    ax.annotate("",xy=(6.25,.95),xytext=(6.25,.46),arrowprops={"arrowstyle":"->","color":CORAL,"lw":1.2})
    ax.text(6.25,.25,"Weak identification: broad posterior or no reconstruction",ha="center",fontsize=7.2,color=CORAL)
    savefig(fig,"fig08_fail_closed_flow")
    # tabular outputs
    write_csv(RES/"synthetic_species.csv",list(species["species_rows"][0].keys()),species["species_rows"])
    write_csv(RES/"synthetic_cv_results.csv",list(cv_rows[0].keys()),cv_rows)
    summary={"metadata":{"synthetic_only":True,"seed":SEED,"n_synthetic_taxa":len(species["clade"]),"n_synthetic_clades":species["n_clades"],"cv_replicates_per_world":90,"uses_empirical_data":False,"uses_RCWA_or_FDTD":False,"phylogeny":"coarse simulated clade labels only","purpose":"software, negative-control and identifiability demonstration"},"scenarios":{}}
    for world in ["optical_signal","phylogeny_confounded","no_direct_signal"]:
        rows=[r for r in cv_rows if r["world"]==world]
        d=np.array([r["delta_r2_spectral_minus_pore"] for r in rows],dtype=float)
        summary["scenarios"][world]={"n_replicates":len(rows),"mean_delta_r2_spectral_minus_pore":float(np.mean(d)),"median_delta_r2_spectral_minus_pore":float(np.median(d)),"q025_delta":float(np.quantile(d,.025)),"q975_delta":float(np.quantile(d,.975)),"interpretation":"synthetic only; not an empirical estimate"}
    (RES/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    spec={"seed":SEED,"data_origin":"generated by verify_model.py; no observed biological or paleo data","surrogate_spectrum":"two Gaussian resonance basis functions multiplied by a bounded geometry factor; not a Maxwell solver","underwater_attenuation":"synthetic wavelength-dependent exponential attenuation with arbitrary coefficients","phylogenetic_control":"synthetic clade fixed intercept; not a molecular phylogeny or Brownian model","validation":"within-clade held-out taxa, OLS, R-squared; 90 synthetic replicates per scenario","scenarios":{"optical_signal":"response depends on surrogate optical score plus random clade effect and noise","phylogeny_confounded":"response depends on clade effect plus noise, not optical score directly","no_direct_signal":"response is noise only"},"limits":["does not estimate real species preference","does not validate phylogenetic comparative method","does not show a paleo-light proxy exists","does not run RCWA/FDTD","does not reconstruct any lake or ocean core"]}
    (RES/"synthetic_generation_spec.json").write_text(json.dumps(spec,indent=2)+"\n",encoding="utf-8")
    report={"status":"PASS","seed":SEED,"checks":[],"limits":"All checks apply to synthetic software pipeline only."}
    def check(name,ok,detail):
        report["checks"].append({"name":name,"status":"PASS" if ok else "FAIL","detail":detail})
        if not ok: raise AssertionError(f"FAIL {name}: {detail}")
    check("surrogate_bounds",bool(np.all((species["T"]>=0)&(species["T"]<=1))),"All surrogate transmission values are within [0,1].")
    for typ in ["clear","mixed","turbid"]:
        arr=attenuation_coeff(wl,typ)
        curve=np.exp(-arr*4)
        check(f"nonnegative_attenuation_{typ}",bool(np.all(arr>=0)),"Synthetic coefficients are nonnegative.")
        check(f"depth_monotonicity_{typ}",bool(np.all(np.exp(-arr*6)<=np.exp(-arr*2)+1e-12)),"For nonnegative K, deeper irradiance is no larger at same wavelength.")
    check("all_three_worlds",{r["world"] for r in cv_rows}=={"optical_signal","phylogeny_confounded","no_direct_signal"},"Positive-signal and negative-control worlds are present.")
    check("heldout_predictions_finite",bool(np.all(np.isfinite([r["r2_spectral_surrogate"] for r in cv_rows]))),"Held-out metrics are finite.")
    check("expected_csvs",all((RES/x).exists() and (RES/x).stat().st_size>20 for x in ["synthetic_species.csv","synthetic_cv_results.csv","attenuation_curves.csv","inverse_identifiability_grid.csv"]),"Data tables needed for plots exist and are nonempty.")
    check("eight_figure_sources",all((FIG/f"fig{i:02d}_{name}.{ext}").exists() for i,name in [(1,"surrogate_spectra"),(2,"attenuation"),(3,"incremental_prediction"),(4,"inverse_nonidentifiability"),(5,"causal_graph"),(6,"feature_graph"),(7,"validation_flow"),(8,"fail_closed_flow")] for ext in ["pdf","png"]),"Eight plot/graph/flow assets were generated in PDF and PNG.")
    report.update({"python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,"networkx":nx.__version__,"figure_count":8,"plot_count":4,"additional_graph_count":2,"flow_diagram_count":2,"empirical_validation":"NOT_PERFORMED"})
    (RES/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (RES/"verification_summary.md").write_text("# Verification summary\n\nStatus: **PASS for declared synthetic-code checks only**.\n\n"+"\n".join(f"- `{c['status']}` — {c['name']}: {c['detail']}" for c in report["checks"])+"\n\nNo empirical biological or paleolimnological validation was performed. The spectral generator is not RCWA/FDTD.\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"checks":len(report["checks"]),"synthetic_only":True,"scenarios":summary["scenarios"]},indent=2))


def main():
    ROOT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True)
    setup_style()
    rng=np.random.default_rng(SEED)
    species=simulate_species(rng)
    cv_rows=cv_scenarios(rng,species)
    make_figures(species,cv_rows)

if __name__=="__main__":
    main()
