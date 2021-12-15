def ZTFAnalysis(starsfull,cluster):
    types = ['CEP','RR']
    Analysis = pd.DataFrame()
    Nreps = 8#
    Percents =  [30,50]

    stats = ['mean','median','std','sem',p16,p84,med_sig,unexplained_variance,Outliers1,Outliers3]
    Stats = ['mean','median','std','sem','p16','p84','med_sig','unexplained_variance', 'Outliers1','Outliers3']
    Widths = np.round(np.logspace(-3,np.log10(0.4),10),3)
    for _type in types:
        stars1 = starsfull[starsfull['Type']==_type].reset_index()
        subs = pd.unique(stars1['Subtype'])
        for _sub in subs:
            stars = stars1[stars1['Subtype']==_sub].reset_index()
            for w in Widths:
                    try:
                        os.mkdir(f'outputs/windows/w_{w}/')
                    except:
                        pass
            for i,row in stars.head(100).iterrows():
                for w in Widths:
                    with open('Bootstrap_log_ZTF.txt','w') as log:
                            print(i, row['source_id'],end='\t',file=log)
                            print(i, row['source_id'],end='\t')
                            filename = f"data/ZTF/{row['source_id']}.csv"
                            D =  pd.read_csv(filename)

                            D['t'] = D['mjd']
                            D['filt'] = D['filtercode']
                            D.to_csv(filename)
                            #for V, d in D.groupby(D['filt']):
                            #    plt.scatter(d['phase'],d['mag'])
                            #plt.show()
                            inds = pd.unique(D['filtercode'])
                            Nmax = D.index.size
                            Sim = MCSimulation("data/ZTF/", row['pf'], filename,verbose=False,memory_type='index')
                            Sim.compute_phase()

                            #for Per in Percents:
                            ok = False


                            for _ in range(1):
                                    Sim.produce_bootstrap(Percents,Nreps)
                                    Sim.remove_window_variable_width(np.array([Widths.size*[0.5],Widths]).T)
                                    print(f"Bootstrap {_}",end='\t',file=log)
                                    print(f"Bootstrap {_}", end='\t')
                                    #Sim.produce_bootstrap([Per],Nreps)
                                    try:
                                        Sim.run_simulation('fast',cluster=cluster)
                                        Sim.Simulated_periods.loc[:,'w'] = np.tile(Widths, np.sort(np.tile(Percents,Nreps)).size)
                                        ok = True
                                    except:
                                        print(Percents,'Fail',end='\t')
                                        print(Percents,'Fail',end='\t',file=log)
                                    if ok:
                                        for w in Widths:

                                                _Sim = Sim.Simulated_periods.loc[Sim.Simulated_periods['w']==w,:]
                                                _Sim.loc[:,'N'] = _Sim['N'].astype(float)/len(inds)
                                                _Sim.loc[:,'Nbin'] = pd.cut(_Sim['N'],[10,20,30,50,np.inf])
                                                Stat = _Sim.groupby('Nbin').agg({'P':stats,'N':'mean'})

                                                for ind in Stat.index:
                                                    Dict = {'Name':row['source_id'],'Gaia period':row['pf'],#'LS period':Sim.best_fitting[1],
                                                    }
                                                    for st in Stats:
                                                        Dict[st] = Stat.loc[ind,('P',st)]
                                                    Dict['window'] = w
                                                    Dict['Nobs'] = ind
                                                    Dict['mean N'] = Stat.loc[ind,('N','mean')]
                                                    Dict['Type'] = _type
                                                    Dict['Subtype'] = _sub
                                                    Analysis = Analysis.append(Dict, ignore_index=True)
                                                    Analysis.to_csv('ManystarsZTF.csv')
                                        print(ok)
                                        print(ok,file=log)

                                        break
    return Analysis