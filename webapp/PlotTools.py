import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl

class PlotTool(object):

    def __init__(self):
        pass

    def plot_fill(self, X, Y, Z, cmap, ax, uniform=True):
        ax.plot(X, Y)
        dx = X[1] - X[0]
        N = X.shape[0]

        if uniform:
            c = np.arange(X.shape[0])
            Zsort = np.argsort(Z)
            Z[Zsort] = c
            den = float(N)
        else:
            den = np.max(Z)

        for n, (x, y, z) in enumerate(zip(X, Y, Z)):
            color = cmap(z / den)
            self.rect(x, 0, dx, y, color, ax)

    def rect(self, x, y, w, h, c, ax):
        polygon = plt.Rectangle((x, y), w, h, color=c)
        ax.add_patch(polygon)

    # def plot_activities(self, analyzer, indices, gradient='norm_speed'):
    #     hills_list = []
    #     for activity in xrange(indices[0], indices[1]):
    #         hills = analyzer.hill_analysis(activity)
    #         if len(hills) != 0:
    #             hills_list.append((activity, hills))


    #     r, c = self.subplot_dims(len(hills_list))
    #     fig, axs = plt.subplots(r, c, figsize=(15,12))

    #     for k, ax in enumerate(axs.reshape(-1)):
    #         dist = analyzer.dist[:, hills_list[k][0]]
    #         alt = analyzer.alt[:, hills_list[k][0]]
    #         dist = analyzer.dist[:, hills_list[k][0]]
    #         self.plot_hills(hills_list[k][0], ax, hills_list[k][1])
    #         ax.legend()

    def get_gradient(gtype):
        if gtype == 'norm_speed':
            pass

    def plot_hills(self, dist, alt, Z, ax, hills, cmap):
        # fig, axs = plt.subplots(1,1, figsize=(12, 8))
        # ax = axs
        # p = PlotTools()
        # plot the altitude profile
        ax.plot(dist, alt, c='r', label='Altitude')
        # ax.set_title(self.activities[ride_idx]['name']+', '+self.activities[ride_idx]['start_date'])
        xmin, xmax = np.min(dist), np.max(dist)
        ymin, ymax = np.min(alt), np.max(alt)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        # fig.colorbar()

        for hill_idx, hill in enumerate(hills):
            label = 'Score: %0.0f, Speed: %0.2f' % (hill['score'], hill['velocity'])
            X = dist[hill['start']:hill['stop']]
            Y = alt[hill['start']:hill['stop']]
            self.plot_fill(X, Y, Z, cmap, ax)
            # ax.plot(X, Y, c='b', linewidth=3, label=label)

        ax.set_ylabel('Altitude (feet)')
        ax.set_xlabel('Distance (mile)')

    def plot_hills2(self, dist, alt, Z, hills, cmap):
        # fig, axs = plt.subplots(1,1, figsize=(12, 8))
        # ax = axs
        # p = PlotTools()
        # plot the altitude profile
        # ax.plot(dist, alt, c='r', label='Altitude')
        # ax.set_title(self.activities[ride_idx]['name']+', '+self.activities[ride_idx]['start_date'])
        xmin, xmax = np.min(dist), np.max(dist)
        ymin, ymax = np.min(alt), np.max(alt)
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([ymin, ymax])
        r, c = self.subplot_dims(len(hills))
        fig, axs = plt.subplots(r, c, figsize=(15, 12))
        
        if r == 1 and c == 1:
            axs = np.array([axs])

        for k, ax in enumerate(axs.reshape(-1)):
            hill = hills[k]
            # cmap = get_cmap()
            label = 'Score: %0.0f, Speed: %0.2f' % (hill['score'], hill['velocity'])
            X = dist[hill['start']:hill['stop']]
            Y = alt[hill['start']:hill['stop']]
            new_Z = Z[hill['start']:hill['stop']]
            self.plot_fill(X, Y, new_Z, cmap, ax)

            ax.set_ylabel('Altitude (feet)')
            ax.set_xlabel('Distance (mile)')


        # for hill_idx, hill in enumerate(hills):
        #     label = 'Score: %0.0f, Speed: %0.2f' % (hill['score'], hill['velocity'])
        #     X = dist[hill['start']:hill['stop']]
        #     Y = alt[hill['start']:hill['stop']]
        #     self.plot_fill(X, Y, Z, cmap, ax)
        #     # ax.plot(X, Y, c='b', linewidth=3, label=label)

        return r, c

    def subplot_dims(self, n):
        if n == 0:
            return (0, 0)
        rows = int(round(np.sqrt(n)))
        cols = int(np.ceil(n/rows))

        return (rows, cols)

if __name__ == '__main__':
    p = PlotTools()
    
    # print p.get_cmap(34)