import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
import matplotlib.tri as tri
import numpy as np
import os

def doAnimationdD(plotGrid, Ufun, FDxGrid, UbenchmarkFun, tTimes, Omega, cmap = [], sortFun = [], sampleFun = [], fname = [], key = [], closePlot = 1):
    dim = Omega.shape[1]

    if(sortFun == []):
        sortFun = lambda x: x
    if(cmap == []):
        cmap = cm.gnuplot2_r

    if(dim == 1):
        plotGrid, FDxGrid, key = sampleFun(0, key)
        plotGrid = plotGrid.reshape((-1,))
        FDxGrid = FDxGrid.reshape((-1,))
        U = jax.vmap(lambda i: Ufun(plotGrid, i), out_axes = 1)(jnp.arange(len(tTimes)))
        Ubenchmark = jnp.asarray([UbenchmarkFun(FDxGrid, i).reshape((-1,)) for i in range(len(tTimes))]).T # jax.vmap(lambda i: UbenchmarkFun(FDxGrid, i).reshape((-1,)), out_axes = 1)(jnp.arange(len(tTimes)))


        Isort = jnp.argsort(plotGrid)
        plotGrid = plotGrid[Isort]
        U = U[Isort, :]

        Isort = jnp.argsort(FDxGrid)
        FDxGrid = FDxGrid[Isort]
        Ubenchmark = Ubenchmark[Isort, :]

        return doAnimation(plotGrid, U, FDxGrid, Ubenchmark, tTimes, knots = [], fname = fname, evalphi = [], alpha = [], closePlot = closePlot), key

        # def sampleUniformHelperPlot(tIter, key):
        #     Nx = 10000
        #     key, subkey = jax.random.split(key)
        #     return jax.random.uniform(subkey, (Nx, Omega.shape[1]), minval = 0., maxval = 1.)*(Omega[1, :] - Omega[0, :]) + Omega[0, :], jax.random.uniform(subkey, (Nx, Omega.shape[1]), minval = 0., maxval = 1.)*(Omega[1, :] - Omega[0, :]) + Omega[0, :], key
        #
        # return doAnimation1DFun(Ufun, UbenchmarkFun, sampleUniformHelperPlot, key, tTimes, fname = fname, closePlot = closePlot), key

    fig, ax = plt.subplots(dim, 2*dim, figsize=(10,5))
    listTruth = [[[] for i in range(dim)] for i in range(dim)]
    listApprox = [[[] for i in range(dim)] for i in range(dim)]

    # plot only the largest 15 percent of points
    if(plotGrid == []):
        curXU, curXUbenchmark, key = sampleFun(0, key)
        Nper = curXU.shape[0]
        if(Nper > 20000):
            Nper = int(jnp.round(Nper)*0.01/(curXU.shape[1]))
        indxI = jnp.asarray(range(0, curXU.shape[0], int(jnp.ceil(curXU.shape[0]/(25*Nper)))))
    else:
        Nper = int(jnp.round(plotGrid.shape[0])*0.01/(plotGrid.shape[1]))
        indxI = jnp.asarray(range(0, plotGrid.shape[0], int(jnp.ceil(plotGrid.shape[0]/(25*Nper)))))
    print('Plotting', Nper, 'points')

    curXU, curXUbenchmark, key = sampleFun(0, key)
    UCur = Ufun(curXU, 0).reshape((-1,))
    UbenchmarkCur = UbenchmarkFun(curXUbenchmark, 0).reshape((-1,))
    IsortU = jnp.argsort(sortFun(UCur))
    IsortUbenchmark = jnp.argsort(sortFun(UbenchmarkCur))
    IsortFDxGrid = jax.vmap(lambda i: jnp.argsort(curXUbenchmark[indxI, i]), out_axes = 1)(jnp.arange(curXUbenchmark.shape[1]))
    IsortPlotGrid = jax.vmap(lambda i: jnp.argsort(curXU[indxI, i]), out_axes = 1)(jnp.arange(curXU.shape[1]))
    vmin, vmax = (jnp.min(UbenchmarkCur[IsortUbenchmark[-Nper:]]), jnp.max(UbenchmarkCur[IsortUbenchmark[-Nper:]]))
    for i in range(dim):
        for j in range(dim):
            if(i == j):
                #Isort = jnp.argsort(FDxGrid[indxI, i])
                listTruth[i][j], = ax[i, j].plot(curXUbenchmark[indxI[IsortFDxGrid[:, i]], i], UbenchmarkCur[indxI[IsortFDxGrid[:, i]]], '-k')
                #Isort = jnp.argsort(plotGrid[indxI, i])
                listApprox[i][j], = ax[i, j + dim].plot(curXU[indxI[IsortPlotGrid[:, i]], i], UCur[indxI[IsortPlotGrid[:, i]]], '-k')

                ax[i, j].set_xlim(Omega[0, i], Omega[1, i])
                ax[i, j + dim].set_xlim(Omega[0, i], Omega[1, i])
                ax[i, j + dim].set_ylim(ax[i, j].get_ylim())
                # ax[i, j].set_title(str(i+1) + ', ' + str(j+1))
                # ax[i, j + dim].set_title(str(i+1) + ', ' + str(j+1))

                if(i != dim-1):
                    ax[i, j].set_xticks([])
                    ax[i, j + dim].set_xticks([])
                if(j != 0):
                    ax[i, j].set_yticks([])
                    ax[i, j + dim].set_yticks([])

                if(i == dim - 1):
                    ax[i, j].set_xlabel('dim ' + str(j+1))
                    ax[i, j + dim].set_xlabel('dim ' + str(j+1))
                continue

            if(i <= j):
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].axis('off')
                ax[i, j + dim].set_xticks([])
                ax[i, j + dim].set_yticks([])
                ax[i, j + dim].axis('off')
                continue

            if(i != dim - 1):
                ax[i, j].set_xticks([])
                ax[i, j + dim].set_xticks([])
            if(j != 0):
                ax[i, j].set_yticks([])
                ax[i, j + dim].set_yticks([])
            if(i == dim -1 and j == 0):
                ax[i, j + dim].set_yticks([])

            #Isort = jnp.argsort(UbenchmarkCur)

            listTruth[i][j] = ax[i, j].scatter(curXUbenchmark[IsortUbenchmark[-Nper:], j], curXUbenchmark[IsortUbenchmark[-Nper:], i], s = 2, marker = 's', c = (UbenchmarkCur[IsortUbenchmark[-Nper:]] - vmin)/(vmax - vmin), cmap = cmap, vmin = 0, vmax = 1)
            #Isort = jnp.argsort(UCur)
            if(jnp.min(UCur[IsortU[-Nper:]]) <= vmax):
                listApprox[i][j] = ax[i, j + dim].scatter(curXU[IsortU[-Nper:], j], curXU[IsortU[-Nper:], i], s = 2, marker = 'P', c=(UCur[IsortU[-Nper:]] - vmin)/(vmax - vmin), cmap = cmap, vmin = 0, vmax = 1)

            ax[i, j].set_xlim(Omega[0, j], Omega[1, j])
            ax[i, j].set_ylim(Omega[0, i], Omega[1, i])
            ax[i, j + dim].set_xlim(Omega[0, j], Omega[1, j])
            ax[i, j + dim].set_ylim(Omega[0, i], Omega[1, i])

            if(i == dim - 1):
                ax[i, j].set_xlabel('dim ' + str(j+1))
                ax[i, j + dim].set_xlabel('dim ' + str(j+1))
            if(j == 0):
                ax[i, j].set_ylabel('dim ' + str(i+1))
                if(i > dim):
                    ax[i, j + dim].set_setylabel('dim ' + str(i+1))

    if(curXU.shape == curXUbenchmark.shape):
        assert(jnp.all(UCur.shape == UbenchmarkCur.shape))
        err = jnp.linalg.norm(UCur - UbenchmarkCur)/jnp.linalg.norm(UbenchmarkCur)
    else:
        err = -1

    ax[0, dim - 1].set_title("Time " + "{:.2e}".format(tTimes[0]) + ", truth", loc='right')
    ax[0, -1].set_title("rel l2 err " + "{:.2e}".format(err), loc='right')

    def animate(tIter, listTruth, listApprox, key):
        print('animate', tIter, len(tTimes))
        curXU, curXUbenchmark, _ = sampleFun(tIter, key) # not randomizing!
        UCur = Ufun(curXU, tIter).reshape((-1,))
        UbenchmarkCur = UbenchmarkFun(curXUbenchmark, tIter).reshape((-1,))
        IsortFDxGrid = jax.vmap(lambda i: jnp.argsort(curXUbenchmark[indxI, i]), out_axes = 1)(jnp.arange(curXUbenchmark.shape[1]))
        IsortPlotGrid = jax.vmap(lambda i: jnp.argsort(curXU[indxI, i]), out_axes = 1)(jnp.arange(curXU.shape[1]))

        IsortU = jnp.argsort(sortFun(UCur))
        IsortUbenchmark = jnp.argsort(sortFun(UbenchmarkCur))

        vmin, vmax = (jnp.min(UbenchmarkCur[IsortUbenchmark[-Nper:]]), jnp.max(UbenchmarkCur[IsortUbenchmark[-Nper:]]))
        for i in range(dim):
            for j in range(dim):
                if(i < j):
                    continue
                if(i == j):
                    #Isort = jnp.argsort(FDxGrid[indxI, i])

                    listTruth[i][j].set_xdata(curXUbenchmark[indxI[IsortFDxGrid[:, i]], i])
                    listTruth[i][j].set_ydata(UbenchmarkCur[indxI[IsortFDxGrid[:, i]]])
                    #listTruth[i][j].remove()
                    #listTruth[i][j], = ax[i, j].plot(FDxGrid[indxI[Isort], i], Ubenchmark[indxI[Isort], tIter], '-k')
                    #Isort = jnp.argsort(plotGrid[indxI, i])
                    #listApprox[i][j].remove()
                    #listApprox[i][j], = ax[i, j + dim].plot(plotGrid[indxI[Isort], i], U[indxI[Isort], tIter], '-k')
                    listApprox[i][j].set_xdata(curXU[indxI[IsortPlotGrid[:, i]], i])
                    listApprox[i][j].set_ydata(UCur[indxI[IsortPlotGrid[:, i]]])

                    ax[i, j].set_xlim(Omega[0, i], Omega[1, i])
                    ax[i, j + dim].set_xlim(Omega[0, i], Omega[1, i])
                    ax[i,j].set_ylim(jnp.min(UbenchmarkCur[indxI[IsortFDxGrid[:, i]]]), jnp.max(UbenchmarkCur[indxI[IsortFDxGrid[:, i]]]))
                    ax[i, j + dim].set_ylim(ax[i, j].get_ylim())
                    ax[i, j].set_title(str(i+1) + ', ' + str(j+1))
                    ax[i, j + dim].set_title(str(i+1) + ', ' + str(j+1))

                    if(i != dim-1):
                        ax[i, j].set_xticks([])
                        ax[i, j + dim].set_xticks([])
                else:
                    listTruth[i][j].set_offsets(jnp.vstack((curXUbenchmark[IsortUbenchmark[-Nper:], j], curXUbenchmark[IsortUbenchmark[-Nper:], i])).T)

                    listTruth[i][j].set_array((UbenchmarkCur[IsortUbenchmark[-Nper:]] - vmin)/(vmax - vmin))


                    #Isort = jnp.argsort(UCur)
                    if(jnp.min(UCur[IsortU[-Nper:]]) <= vmax):
                        if(listApprox[i][j] == []): # if we plot the approximation the first time
                            listApprox[i][j] = ax[i, j + dim].scatter(curXU[IsortU[-Nper:], j], curXU[IsortU[-Nper:], i], s = 2, marker = 'P', c=(UCur[IsortU[-Nper:]] - vmin)/(vmax - vmin), cmap = cmap, vmin = 0, vmax = 1)
                        else:
                            listApprox[i][j].set_offsets(jnp.vstack((curXU[IsortU[-Nper:], j], curXU[IsortU[-Nper:], i])).T)
                            listApprox[i][j].set_array((UCur[IsortU[-Nper:]] - vmin)/(vmax - vmin))

                        # listApprox[i][j] = ax[i, j + dim].scatter(plotGrid[Isort[-Nper:], j], plotGrid[Isort[-Nper:], i], s = 2, marker = 'P', c=U[Isort[-Nper:], tIter], cmap = cm.plasma_r, vmin = vmin, vmax = vmax)

                    ax[i, j].set_xlim(Omega[0, j], Omega[1, j])
                    ax[i, j].set_ylim(Omega[0, i], Omega[1, i])
                    ax[i, j + dim].set_xlim(Omega[0, j], Omega[1, j])
                    ax[i, j + dim].set_ylim(Omega[0, i], Omega[1, i])

                    # ax[i, j].set_title(str(i+1) + ', ' + str(j+1))
                    # ax[i, j + dim].set_title(str(i+1) + ', ' + str(j+1))

        if(curXU.shape == curXUbenchmark.shape):
            err = jnp.linalg.norm(UCur - UbenchmarkCur)/jnp.linalg.norm(UbenchmarkCur)
        else:
            err = -1
        ax[0, dim - 1].set_title("Time " + "{:.2e}".format(tTimes[tIter]) + ", truth", loc='right')
        ax[0, -1].set_title("rel l2 err " + "{:.2e}".format(err), loc='right')

    anim = FuncAnimation(fig, animate, fargs=(listTruth, listApprox, key), frames=len(tTimes), interval=200, blit=False, repeat=False)

    if(fname != []):
        if(len(tTimes) == 1):
            if(fname[-4:] == '.pdf' or fname[-4:] == '.png'):
                print('Write', fname)
                plt.savefig(fname)
            else:
                print('Write', fname + ".pdf")
                plt.savefig(fname + ".pdf")
        else:
            print("Write", fname + ".mp4")
            anim.save(fname + ".mp4")

    if(closePlot):
        plt.close(fig)
        return None, key
    else:
        return anim, key


def doAnimation(plotGrid, U, FDxGrid, Ubenchmark, tTimes, knots = [], fname = [], evalphi = [], alpha = [], closePlot = 1):
    tolForZoom = 1e-10

    fig, (ax, axZoom) = plt.subplots(1, 2)  # Create a figure containing a single axes.
    fig.set_size_inches(10, 5)
    if(knots != []):
        N = knots.shape[0]
        Ncolors = N + N % 2
        ccc = jnp.linspace(0, 1, Ncolors).reshape((2,-1)).T.reshape((-1,))
        ccc = ccc[:N]
        pathKnots = ax.scatter(knots[:, 0], jnp.zeros((N, )), c = ccc, label = 'knots', zorder = 1)
        pathKnotsZoom = axZoom.scatter(knots[:, 0], jnp.zeros((N, )), c = ccc, label = 'knots', zorder = 1)
    lineNG, = ax.plot(plotGrid, U[:, 0], '-', label='predict', linewidth = 2)  # Plot some data on the axes.
    lineNGZoom, = axZoom.plot(plotGrid, U[:, 0], '-', label='predict', linewidth = 2)  # Plot some data on the axes.
    lineTruth, = ax.plot(FDxGrid, Ubenchmark[:, 0], ':', label = 'truth', linewidth = 2)
    lineTruthZoom, = axZoom.plot(FDxGrid, Ubenchmark[:, 0], ':', label = 'truth', linewidth = 2)
    err = jnp.linalg.norm(np.interp(FDxGrid.ravel(), plotGrid.ravel(), U[:, 0]) - Ubenchmark[:, 0].ravel())/jnp.linalg.norm(Ubenchmark[:, 0].ravel())

    # plot weights
    if(knots != [] and alpha != []):
        lineKnots = ax.plot(jnp.vstack((knots[:, 0], knots[:, 0])), jnp.vstack((jnp.zeros((N,)), alpha[:, 0])), '-r')
        lineKnotsZoom = axZoom.plot(jnp.vstack((knots[:, 0], knots[:, 0])), jnp.vstack((jnp.zeros((N,)), alpha[:, 0])), '-r')
        lineWeights, = ax.plot(knots[:, 0], alpha[:, 0], 'xr', label = 'weights')
        lineWeightsZoom, = axZoom.plot(knots[:, 0], alpha[:, 0], 'xr', label = 'weights')

    # plot unit shapes
    if(alpha != [] and evalphi != []):
        N = alpha.shape[0]
        lineUnits = ax.plot(plotGrid, jnp.dot(evalphi[:, :, 0].reshape((-1,N)), jnp.diag(alpha[:, 0])), '-k', alpha=0.3)
        lineUnitsZoom = axZoom.plot(plotGrid, jnp.dot(evalphi[:, :, 0].reshape((-1,N)), jnp.diag(alpha[:, 0])), '-k', alpha=0.3)

    axZoom.set_xlim(findSupportHelper(Ubenchmark[:, 0], FDxGrid, tolForZoom))
    ax.set_xlabel('spatial domain')
    axZoom.set_xlabel('spatial domain')
    ax.set_ylabel('numerical solution')
    axZoom.set_ylabel('numerical solution')
    plt.legend()
    plt.title("Time " + "{:.2e}".format(tTimes[0]) + ", rel l2 err " + "{:.5e}".format(err))

    def animate(tIter):
        lineNG.set_ydata(U[:, tIter])
        lineNGZoom.set_ydata(U[:, tIter])
        lineTruth.set_ydata(Ubenchmark[:, tIter])
        lineTruthZoom.set_ydata(Ubenchmark[:, tIter])


        if(knots != []):
            pathKnots.set_offsets(jnp.hstack((knots[:, tIter].reshape((N,1)), jnp.zeros((N,1)))))
            pathKnotsZoom.set_offsets(jnp.hstack((knots[:, tIter].reshape((N,1)), jnp.zeros((N,1)))))
        if(knots != [] and alpha != []):
            for lkIter in range(len(lineKnots)):
                lk = lineKnots[lkIter]
                lk.set_xdata(jnp.vstack((knots[lkIter, tIter], knots[lkIter, tIter])))
                lk.set_ydata(jnp.vstack((jnp.zeros((1,)), alpha[lkIter, tIter])))
            for lkIter in range(len(lineKnotsZoom)):
                lk = lineKnotsZoom[lkIter]
                lk.set_xdata(jnp.vstack((knots[lkIter, tIter], knots[lkIter, tIter])))
                lk.set_ydata(jnp.vstack((jnp.zeros((1,)), alpha[lkIter, tIter])))
            lineWeights.set_xdata(knots[:, tIter])
            lineWeights.set_ydata(alpha[:, tIter])
            lineWeightsZoom.set_xdata(knots[:, tIter])
            lineWeightsZoom.set_ydata(alpha[:, tIter])
        if(alpha != [] and evalphi != []):
            for lkIter in range(len(lineUnits)):
                lk = lineUnits[lkIter]
                lk.set_ydata(evalphi[:, lkIter, tIter].reshape((-1,))*alpha[lkIter, tIter])
            for lkIter in range(len(lineUnitsZoom)):
                lk = lineUnitsZoom[lkIter]
                lk.set_ydata(evalphi[:, lkIter, tIter].reshape((-1,))*alpha[lkIter, tIter])
        # find first and last indices of benchmark that is great tolerance
        axZoom.set_xlim(findSupportHelper(Ubenchmark[:, tIter], FDxGrid, tolForZoom))
        lowLim = min(jnp.hstack((U[:, tIter], Ubenchmark[:, tIter])))
        maxLim = max(jnp.hstack((U[:, tIter], Ubenchmark[:, tIter])))
        if(lowLim < 0):
            lowLim = lowLim*1.1
        else:
            lowLim = lowLim*0.9
        if(maxLim < 0):
            maxLim = maxLim*0.9
        else:
            maxLim = maxLim*1.1
        ax.set_ylim(lowLim, maxLim)
        axZoom.set_ylim(ax.get_ylim())
        # axZoom.set_xlim([10, 12])
        err = jnp.linalg.norm(np.interp(FDxGrid.ravel(), plotGrid.ravel(), U[:, tIter].ravel()) - Ubenchmark[:, tIter].ravel())/jnp.linalg.norm(Ubenchmark[:, tIter].ravel())
        ax.set_title("Time " + "{:.2e}".format(tTimes[tIter]) + ", rel l2 err " + "{:.5e}".format(err))
        axZoom.set_title("Time " + "{:.2e}".format(tTimes[tIter]) + ", rel l2 err " + "{:.5e}".format(err))


    anim = FuncAnimation(fig, animate, frames=len(tTimes), interval=200, blit=False, repeat=False)
    if(fname != []):
        if(len(tTimes) == 1):
            print('Write', fname)
            plt.savefig(fname)
        else:
            print("Write", fname + ".mp4")
            anim.save(fname + ".mp4")
    if(closePlot):
        plt.close(fig)
    else:
        return anim

def doCombinedSpaceTimePlot(dataList, fname, xlabelList = [], ylabelList = [], rowslabelList = [], fontSize = 14, plotICFT = [], hspace = 0.1, wspace = 0.1, xspace = 12.8*2.0, yspace = 4.8*2.0, vmin = [], vmax = []):
    # get number of rows (number of data sets)
    nrRows = len(dataList)
    # plot initial condition and final time
    nrICFT = 0
    if(plotICFT != []):
        nrICFT = 1
    # get number of columns (dimensions)
    nrCols = dataList[0][0].shape[0]

    fig, axs = plt.subplots(nrRows + nrICFT, nrCols, figsize=(xspace, yspace))
    if(nrRows + nrICFT == 1):
        axs = axs.reshape((1, -1))
    for i in range(nrRows):
        for j in range(nrCols):
            U = dataList[i][0][j, :, :].reshape((dataList[i][0].shape[1], dataList[i][0].shape[2]))
            plotGrid = dataList[i][1][j, :]
            tTime = dataList[i][3]
            Omega = dataList[i][2][:, j]
            if(vmin != [] and vmax != []):
                axs[i, j].pcolormesh(plotGrid.reshape((-1,)), jnp.asarray(tTime).reshape((-1,)), U.T, cmap = cm.coolwarm, linewidth = 0, shading = 'auto', rasterized = True, vmin = vmin, vmax = vmax)
            else:
                axs[i, j].pcolormesh(plotGrid.reshape((-1,)), jnp.asarray(tTime).reshape((-1,)), U.T, cmap = cm.coolwarm, linewidth = 0, shading = 'auto', rasterized = True)

            if(i < nrRows - 1):
                axs[i, j].axes.xaxis.set_visible(False)
            else:
                if(xlabelList == []):
                    xlabel = 'spatial domain'
                elif(isinstance(xlabelList, list)):
                    xlabel = xlabelList[j]
                else:
                    xlabel = xlabelList
                axs[i, j].set_xlabel(xlabel, size = fontSize)
                axs[i, j].set_xlim(Omega[0], Omega[1])
                axs[i, j].tick_params(axis='x', which='major', labelsize=fontSize)
                axs[i, j].tick_params(axis='x', which='minor', labelsize=fontSize)
            if(j > 0):
                axs[i, j].axes.yaxis.set_visible(False)
            else:
                if(ylabelList == []):
                    ylabel = 'time'
                elif(isinstance(ylabelList, list)):
                    ylabel = ylabelList[i]
                else:
                    ylabel = ylabelList
                axs[i, j].set_ylabel(ylabel, size = fontSize)
                axs[i, j].set_ylim(tTime[0], tTime[-1])
                axs[i, j].tick_params(axis='y', which='major', labelsize=fontSize)
                axs[i, j].tick_params(axis='y', which='minor', labelsize=fontSize)


            if(rowslabelList != []):
                if(j == 0):
                    xmin, xmax = axs[i, j].get_xlim()
                    ymin, ymax = axs[i, j].get_ylim()

                    axs[i, j].text(xmax, ymax, rowslabelList[i], fontsize = fontSize, horizontalalignment = 'right', verticalalignment = 'top', bbox=dict(facecolor='white', alpha=1.0))


    if(nrICFT > 0):
        vmin = []
        vmax = []
        for j in range(nrCols):
            UIC = dataList[-1][0][j, :, 0].reshape((-1,))
            UFT = dataList[-1][0][j, :, -1].reshape((-1,))
            plotGrid = dataList[-1][1][j, :]
            Omega = dataList[-1][2][:, j]
            axs[-1, j].plot(plotGrid, UIC, '-', linewidth = 2, color = 'tab:green', label='initial')  # Plot some data on the axes.
            axs[-1, j].plot(plotGrid, UFT, ':', linewidth = 2, color = 'tab:orange', label = 'final')
            axs[-1, j].plot(plotGrid, plotICFT[j, :].reshape((-1,)), '--k', linewidth = 2, label = 'truth')
            plt.rc('legend', fontsize=fontSize)    # legend fontsize
            axs[-1, j].legend()
            # plt.title("Time " + str(tTime) + ", rel l2 err " + "{:.5e}".format(err))
            axs[-1, j].set_xlabel('dimension ' + str(j + 1), size = fontSize)
            axs[-1, j].set_ylabel('numerical solution', size = fontSize)
            axs[-1, j].tick_params(axis='both', which='major', labelsize=fontSize)
            axs[-1, j].tick_params(axis='both', which='minor', labelsize=fontSize)
            axs[-1, j].set_xlim(Omega[0], Omega[1])
            axs[-1, j].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            #axs[-1, j].yaxis.set_major_formatter(FormatStrFormatter('%d'))
            vminTmp, vmaxTmp = axs[-1, j].get_ylim()
            if(vmin == [] and vmax == []):
                vmin = vminTmp
                vmax = vmaxTmp
            if(vminTmp <= vmin):
                vmin = vminTmp
                for k in range(j):
                    axs[-1, k].set_ylim(vmin, vmax)
            if(vmaxTmp >= vmax):
                vmax = vmaxTmp
                for k in range(j):
                    axs[-1, k].set_ylim(vmin, vmax)
            if(j > 0):
                axs[-1, j].set_ylim(vmin, vmax)
                axs[-1, j].axes.yaxis.set_visible(False)




    plt.subplots_adjust(hspace = hspace, wspace = wspace)
    fig.subplots_adjust(bottom=0.2)

    #plt.tight_layout()
    plt.savefig(fname)
    os.system('pdfcrop ' + fname)
    plt.close(fig)


def doSpaceTimePlot(plotGrid, U, tTime, Omega, fname, vmin = [], vmax = [], xlabel = [], ylabel = [], rasterize = True, fontSize = 22, showColorBar = False, colorBarArgs = {}, ylabelpad = None, xticks = None):
    if(showColorBar):
        # fig, (cax, ax) = plt.subplots(nrows = 2, gridspec_kw={'height_ratios': [0.05, 1]})
        #fig, (ax, cax) = plt.subplots(ncols = 2, figsize = (6.4 + 6.4*0.15, 4.8), gridspec_kw={'width_ratios': [1, 0.15]})
        fig, ax = plt.subplots(figsize = (6.4 + 6.4*0.15 + 0.05, 4.8))
    else:
        fig, ax = plt.subplots(figsize = (6.4, 4.8))

    plt.rc('font', size=fontSize)          # controls default text sizes
    plt.rc('axes', titlesize=fontSize)     # fontsize of the axes title
    plt.rc('axes', labelsize=fontSize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontSize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontSize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fontSize)    # legend fontsize
    plt.rc('figure', titlesize=fontSize)  # fontsize of the figure title
    if(vmin == [] or vmax == []):
        fplot = ax.pcolormesh(jnp.asarray(tTime).reshape((-1,)), plotGrid.reshape((-1,)), U, cmap = cm.coolwarm, linewidth = 0, shading = 'auto', rasterized = rasterize)
    else:
        fplot = ax.pcolormesh(jnp.asarray(tTime).reshape((-1,)), plotGrid.reshape((-1,)), U, cmap = cm.coolwarm, linewidth = 0, shading = 'auto', vmin = vmin, vmax = vmax, rasterized = rasterize)
    #ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    if(showColorBar):
        fig.colorbar(fplot, ax = ax, **colorBarArgs) #, orientation="horizontal")

    if(xlabel == []):
        xlabel = 'time'
    ax.set_xlabel(xlabel, size = fontSize)
    if(ylabel == []):
        ylabel = 'spatial domain'
    ax.set_ylabel(ylabel, size = fontSize, labelpad = ylabelpad)
    ax.tick_params(axis='both', which='major', labelsize=fontSize)
    ax.tick_params(axis='both', which='minor', labelsize=fontSize)
    if(xticks):
        ax.set_xticks(ticks = xticks[0])
        ax.set_xticklabels(labels = xticks[1])
    ax.set_ylim(Omega[0], Omega[1])
    ax.set_xlim(tTime[0], tTime[-1])

    if(vmin == [] or vmax == []):
        vmin, vmax = fplot.get_clim()

    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)

    return vmin, vmax


def doPlot2D(plotGrid, U, FDxGrid, Ubenchmark, tTime, knots = [], fname = [], evalphi = [], alpha = [], closePlot = 1):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    if(knots != []):
        N = knots.shape[0]
        Ncolors = N + N % 2
        ccc = jnp.linspace(0, 1, Ncolors).reshape((2,-1)).T.reshape((-1,))
        ccc = ccc[:N]
        ax.scatter(knots, jnp.zeros((N, )), c = ccc, label = 'knots', zorder = 1)

    oneDimNFD = int(np.sqrt(FDxGrid.shape[0]))
    fdX = FDxGrid[:, 0].reshape((oneDimNFD, oneDimNFD))
    fdY = FDxGrid[:, 1].reshape((oneDimNFD, oneDimNFD))
    oneDimN = int(np.sqrt(plotGrid.shape[0]))
    X = plotGrid[:, 0].reshape((oneDimNFD, oneDimNFD))
    Y = plotGrid[:, 1].reshape((oneDimNFD, oneDimNFD))
    #ax.plot_surface(fdX, fdY, Ubenchmark.reshape((oneDimN, oneDimN)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, U.reshape((oneDimN, oneDimN)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    if(plotGrid.shape == FDxGrid.shape):
        err = jnp.linalg.norm(U.ravel() - Ubenchmark.ravel())/jnp.linalg.norm(Ubenchmark)
    else:
        err = -1

    plt.legend()
    plt.title("Time " + str(tTime) + ", rel l2 err " + "{:.5e}".format(err))
    plt.xlabel('x1')
    plt.ylabel('x2')

    if(fname != []):
        print("write", fname)
        plt.savefig(fname)
    if(closePlot):
        plt.close()

def doPlot(plotGrid, U, FDxGrid, Ubenchmark, tTime, knots = [], fname = [], evalphi = [], alpha = [], closePlot = 1):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    if(knots != []):
        N = knots.shape[0]
        Ncolors = N + N % 2
        ccc = jnp.linspace(0, 1, Ncolors).reshape((2,-1)).T.reshape((-1,))
        ccc = ccc[:N]
        ax.scatter(knots, jnp.zeros((N, )), c = ccc, label = 'knots', zorder = 1)
    ax.plot(plotGrid, U, '-', label='fitted IC')  # Plot some data on the axes.
    ax.plot(FDxGrid, Ubenchmark, ':', label = 'truth')
    if(plotGrid.shape == FDxGrid.shape):
        err = jnp.linalg.norm(U.ravel() - Ubenchmark.ravel())/jnp.linalg.norm(Ubenchmark)
    else:
        err = -1

    # plot weights
    if(knots != [] and alpha != []):
        ax.plot(jnp.vstack((knots, knots)), jnp.vstack((jnp.zeros((N,)), alpha)), '-r')
        ax.plot(knots, alpha, 'xr', label = 'weights')

    # plot unit shapes
    if(alpha != [] and evalphi != []):
        ax.plot(plotGrid, jnp.dot(evalphi, jnp.diag(alpha)), '-k', alpha=0.3)

    plt.legend()
    plt.title("Time " + str(tTime) + ", rel l2 err " + "{:.5e}".format(err))
    plt.xlabel('spatial domain')
    plt.ylabel('numerical solution')

    if(fname != []):
        print("write", fname)
        plt.savefig(fname)
    if(closePlot):
        plt.close()

def findSupportHelper(U, plotGrid, tol):
    imin = jnp.argmax(U>tol)
    imax = U.size - jnp.argmax(jnp.flip(U) > tol) - 1
    return plotGrid[imin], plotGrid[imax]
