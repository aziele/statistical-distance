"""
Distance measures to compare two probability density functions (pdfs).
"""

import numpy as np
import numpy.ma as ma


class Distance:

    def __init__(self, epsilon=None):
        self.epsilon = np.finfo(float).eps if not epsilon else epsilon

    def acc(self, u, v):
        """
        The average of Manhattan and Chebyshev distances.

        Synonyms: 
            ACC distance
            Average distance

        References:
            1. Krause EF (2012) Taxicab Geometry An Adventure in Non-Euclidean 
               Geometry. Dover Publications.
            2. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               vol. 1(4), pp. 300-307.

        """
        return (self.manhattan(u, v) + self.chebyshev(u, v))/2

    def add_chisq(self, u, v):
        """
        Additive Symmetric Chi-square distance.

        References:
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               vol. 1(4), pp. 300-307.

        """
        uvmult = u * v
        with np.errstate(divide='ignore', invalid="ignore"):
            return np.sum(np.where(uvmult != 0, ((u-v)**2 * (u+v))/uvmult, 0))     


    def bhattacharyya(self, u, v):
        """
        Bhattacharyya distance.

        Returns a distance value between 0 and 1.

        References:
            1. Bhattacharyya A (1947) On a measure of divergence between two 
               statistical populations defined by probability distributions, 
               Bull. Calcutta Math. Soc., 35, 99–109.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
            3. https://en.wikipedia.org/wiki/Bhattacharyya_distance

        """
        return -np.log(np.sum(np.sqrt(u*v)))

    def braycurtis(self, u, v):
        """
        Bray-Curtis distance.

        Synonyms:
            Sørensen distance
            Bray-Curtis dissimilarity
        
        Note:
            When used for comparing two probability density functions (pdfs),
            Bray-Curtis distance equals Manhattan distance divided by 2.

        References:
            1. Bray JR, Curtis JT (1957) An ordination of the upland forest of
               the southern Winsconsin. Ecological Monographies, 27, 325-349.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
            3. https://en.wikipedia.org/wiki/Bray–Curtis_dissimilarity

        """
        return np.sum(np.abs(u - v)) / np.sum(np.abs(u + v))

    def canberra(self, u, v):
        """
        Canberra distance.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        try:
            d = np.nansum(np.abs(u - v) / (np.abs(u) + np.abs(v)))
        finally:
            np.seterr(invalid='ignore')
        return d

    def chebyshev(self, u, v):
        """
        Chebyshev distance.

        Synonyms:
            Chessboard distance
            King-move metric
            Maximum value distance
            Minimax approximation

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.        
        """
        return np.amax(np.abs(u - v))

    def chebyshev_min(self, u, v):
        """
        Minimum value distance (my measure).

        """
        return np.amin(np.abs(u - v))

    def clark(self, u, v):
        """
        Clark distance.

        Clark distance equals the squared root of half of the divergence.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return np.sqrt(np.nansum(np.power(np.abs(u-v)/(u+v),2)))

    # def correlation_pearson(self, u, v):
    #     """
    #     Pearson correlation distance.

    #     Returns a distance value between 0 and 2.

    #     """
    #     r = ma.corrcoef(u, v)[0, 1]
    #     return 1.0 - r

    # def correlation(self, u, v):
    #     """
    #     Correlation distance.
    #     """
    #     # TODO: fix nan when diveded by 0 and no std
    #     umu = u.mean()
    #     vmu = v.mean()
    #     um = u - umu
    #     vm = v - vmu
    #     return 1.0 - np.dot(um, vm) / (np.linalg.norm(um)*np.linalg.norm(vm))

    def cosine(self, u, v):
        """
        Cosine distance.

        References:
            1. SciPy.

        """
        return 1 - np.dot(u, v)/(np.sqrt(np.dot(u, u))*np.sqrt(np.dot(v, v)))

    def czekanowski(self, u, v):
        """
        Czekanowski distance.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return np.sum(np.abs(u - v)) / np.sum(u + v)

    def dice(self, u, v):
        """
        Dice dissimilarity.

        Synonyms:
            Sorensen distance

        Referemces:
            1. Dice LR (1945) Measures of the amount of ecologic association
               between species. Ecology. 26, 297-302.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        u_v = u - v
        return np.dot(u_v, u_v) / (np.dot(u, u) + np.dot(v, v))

    def divergence(self, u, v):
        """
        Divergence.

        Divergence equals squared Clark distance multiplied by 2.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return 2 * np.nansum(np.power(u-v,2) / np.power(u+v,2))

    def euclidean(self, u, v):
        """
        Euclidean distance.

        Synonyms:
            Pythagorean metric

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.        
        """
        return np.linalg.norm(u-v)

    def google(self, u, v):
        """
        Normalized Google Distance (NGD).

        Returns a distance value between 0 and 1. Two sequences are treated
        as two different web pages and the each word frequency represents 
        terms found in each webpage.

        Note:
            When used for comparing two probability density functions (pdfs),
            Google distance equals half of Manhattan distance.

        References:
            1. Lee & Rashid (2008) Information Technology, ITSim 2008.
               doi:10.1109/ITSIM.2008.4631601.

        """
        x = float(np.sum(u))
        y = float(np.sum(v))
        summin = float(np.sum(np.minimum(u, v)))
        return (max([x, y]) - summin) / ((x + y) - min([x, y]))

    def gower(self, u, v):
        """
        Gower distance.

        Gower distance equals Manhattan distance divided by vector length.

        References:
            1. Gower JC. (1971) General Coefficient of Similarity
               and Some of Its Properties, Biometrics 27, 857-874.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return np.sum(np.abs(u - v)) / u.size

    def hellinger(self, u, v):
        """
        Hellinger distance.

        Note:
            This implementation produces values two times larger than values
            obtained by Hellinger distance described in Wikipedia and also
            in https://gist.github.com/larsmans/3116927.

            Wikipedia:
            np.sqrt(np.sum((np.sqrt(u) - np.sqrt(v)) ** 2)) / np.sqrt(2) 

        References:
           1.  Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return np.sqrt(2*np.sum((np.sqrt(u) - np.sqrt(v)) ** 2))

    def jaccard(self, u, v):
        """
        Jaccard distance.

        References:
           1.  Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        uv = np.dot(u, v)
        return 1 - (uv / (np.dot(u, u) + np.dot(v, v) - uv))

    def jeffreys(self, u, v):
        """
        Jeffreys divergence.

        Synonyms:
            J divergence

        References:
            1. Jeffreys H (1946) An Invariant Form for the Prior Probability
               in Estimation Problems. Proc.Roy.Soc.Lon., Ser. A 186, 453-461.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        # Add epsilon to zeros in vectors to avoid division
        # by 0 and/or log of 0. Alternatively, zeros in the
        # vectors could be ignored or masked (see below).
        # u = ma.masked_where(u == 0, u)
        # v = ma.masked_where(v == 0, u)
        u = np.where(u==0, self.epsilon, u)
        v = np.where(v==0, self.epsilon, v)
        return np.sum((u-v) * np.log(u / v))

    def jensenshannon_divergence(self, u, v):
        """
        Jensen-Shannon divergence.

        Note:
            1. Equals half of Topsøe distance
            2. Equals squared jensenshannon_distance.

        References:
            1. Lin J. (1991) Divergence measures based on the Shannon entropy.
               IEEE Transactions on Information Theory, 37(1):145–151.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        Comments:
            Equals Jensen difference in Sung-Hyuk (2007):
            u = np.where(u==0, self.epsilon, u)
            v = np.where(v==0, self.epsilon, v)
            el1 = (u * np.log(u) + v * np.log(v)) / 2
            el2 = (u + v)/2
            el3 = np.log(el2)
            return np.sum(el1 - el2 * el3)          

        """
        u = np.where(u==0, self.epsilon, u)
        v = np.where(v==0, self.epsilon, v)        
        dl = u * np.log(2*u/(u+v))
        dr = v * np.log(2*v/(u+v))
        return (np.sum(dl) + np.sum(dr)) / 2

    def k_divergence(self, u, v):
        """
        K divergence.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        u = np.where(u==0, self.epsilon, u)
        v = np.where(v==0, self.epsilon, v)
        return np.sum(u*np.log(2*u/(u+v)))

    def kl_divergence(self, u, v):
        """
        Kullback-Leibler divergence.

        Syonymes:
            KL divergence, relative entropy, information deviation

        References:
            1. Kullback S, Leibler RA (1951) On information and sufficiency.
               Ann. Math. Statist. 22:79–86
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        """
        u = np.where(u==0, self.epsilon, u)
        v = np.where(v==0, self.epsilon, v)
        return np.sum(u * np.log(u / v))

    def kulczynski(self, u, v):
        """
        Kulczynski distance.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        """
        return np.sum(np.abs(u - v)) / np.sum(np.minimum(u, v))

    def kumarjohnson(self, u, v):
        """
        Kumar-Johnson distance.

        References:
            1. Kumar P, Johnson A. (2005) On a symmetric divergence measure
               and information inequalities, Journal of Inequalities in pure
               and applied Mathematics. 6(3).
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        """
        uvmult = u*v
        with np.errstate(divide='ignore'):
            numer = np.power(u**2 - v**2, 2)
            denom = 2 * np.power(uvmult, 3/2)
            return np.sum(np.where(uvmult != 0, numer/denom, 0))

    def lorentzian(self, u, v):
        """
        Lorentzian distance.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.        

        Note:
            One (1) is added to guarantee the non-negativity property and to 
            eschew the log of zero

        """
        return np.sum(np.log(np.abs(u-v)+1))

    def manhattan(self, u, v):
        """
        Manhattan distance.

        Synonyms:
            City block distance
            Rectilinear distance
            Taxicab norm

        Notes:
            Manhattan distance between two probability density functions 
            (pdfs) equals:
            1. Non-intersection distance multiplied by 2.
            2. Gower distance multiplied by vector length.
            3. Bray-Curtis distance multiplied by 2.
            4. Google distance multiplied by 2.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        """
        return np.sum(np.abs(u - v))

    def marylandbridge(self, u, v):
        """
        Maryland Bridge distance.

        References:
            1. Deza M, Deza E (2009) Encyclopedia of Distances. 
               Springer-Verlag Berlin Heidelberg. 1-590.

        """
        uvdot = np.dot(u, v)
        return 1 - (uvdot/np.dot(u, u) + uvdot/np.dot(v, v))/2

    def matusita(self, u, v):
        """
        Matusita distance.

        Notes:
            Equals square root of Squared-chord distance.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        """
        return np.sqrt(2 - 2*np.sum(np.sqrt(u*v)))

    def minkowski(self, u, v, p=2):
        """
        Minkowski distance.

        Parameters:
            p : int
                The order of the norm of the difference.

        Notes:
            When p goes to infinite, the Chebyshev distance is derived.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4):300-307.

        """
        return np.linalg.norm(u - v, ord=p)

    def motyka(self, u, v):
        """
        Motyka distance.

        Notes:
            The distance between identical vectors is not equal to 0 but 0.5.
        
        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return np.sum(np.maximum(u, v)) / np.sum(u + v)

    def neyman_chisq(self, u, v):
        """
        Neyman chi-square distance.

        References:
            1. Neyman J (1949) Contributions to the theory of the chi^2 test. 
               In Proceedings of the First Berkley Symposium on Mathematical
               Statistics and Probability.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        with np.errstate(divide='ignore'):
           return np.sum(np.where(u != 0, (u-v)**2/u, 0))

    def nonintersection(self, u, v):
        """
        Distance based on intersection.

        Synonyms:
            Non-overlaps
            Intersection distance

        Notes:
            The distance between two pdfs is Manhattan distance divded by 2.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return 1 - np.sum(np.minimum(u, v))

    def pearson_chisq(self, u, v):
        """
        Pearson chi-square divergence.

        Notes:
            Pearson chi-square divergence is asymmetric.

        References:
            1. Pearson K. (1900) On the Criterion that a given system of 
               deviations from the probable in the case of correlated system
               of variables is such that it can be reasonable supposed to have
               arisen from random sampling, Phil. Mag. 50, 157-172.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        """
        with np.errstate(divide='ignore'):
            return np.sum(np.where(v != 0, (u-v)**2/v, 0))

    def penroseshape(self, u, v):
        """
        Penrose shape distance.

        References:
            1. Deza M, Deza E (2009) Encyclopedia of Distances. 
               Springer-Verlag Berlin Heidelberg. 1-590.

        """
        umu = np.mean(u)
        vmu = np.mean(v)
        return np.sqrt(np.sum(((u-umu)-(v-vmu))**2))

    def soergel(self, u, v):
        """
        Soergel distance.


        Notes:
            Equals Tanimoto distance.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return np.sum(np.abs(u - v)) / np.sum(np.maximum(u, v))

    def squared_chisq(self, u, v):
        """
        Squared chi-square distance.

        Synonyms:
            Triangular discrimination

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        uvsum = u + v
        with np.errstate(divide='ignore'):
            return np.sum(np.where(uvsum != 0, (u-v)**2/uvsum, 0))


    def squaredchord(self, u, v):
        """
        Squared-chord distance.

        Notes:
            Equals to squared Matusita distance.

        Reference:
            1. Gavin DG et al. (2003) A statistical approach to evaluating 
               distance metrics and analog assignments for pollen records.
               Quaternary Research 60:356–367.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.
        
        """
        return np.sum((np.sqrt(u) - np.sqrt(v))**2)

    def sqeuclidean(self, u, v):
        """
        Squared Euclidean distance.

        Notes:
            Equals to squared Euclidean distance.

        Reference:
            1. Gavin DG et al. (2003) A statistical approach to evaluating 
               distance metrics and analog assignments for pollen records.
               Quaternary Research 60:356–367.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        return np.dot((u - v), (u - v))

    def taneja(self, u, v):
        """
        Taneja distance.

        References:
            1. Taneja IJ. (1995), New Developments in Generalized Information
               Measures, Chapter in: Advances in Imaging and Electron Physics,
               Ed. P.W. Hawkes, 91, 37-135.
            2. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307.

        """
        u = np.where(u==0, self.epsilon, u)
        v = np.where(v==0, self.epsilon, v)
        uvsum = u + v
        return np.sum((uvsum/2)*np.log(uvsum/(2*np.sqrt(u*v))))

    def tanimoto(self, u, v):
        """
        Tanimoto distance.

        Notes:
            Equals Soergel distance.

        References:
            1. Sung-Hyuk C. (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307

        """
        #return np.sum(abs(u-v)) / np.sum(np.maximum(u, v))
        usum = np.sum(u)
        vsum = np.sum(v)
        minsum = np.sum(np.minimum(u, v))
        return (usum + vsum - 2*minsum) / (usum + vsum - minsum)

    def topsoe(self, u, v):
        """
        Topsøe distance.

        Synonyms:
            Information statistic

        Notes:
            Equals two times Jensen-Shannon divergence.

        References:
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307

        """
        u = np.where(u==0, self.epsilon, u)
        v = np.where(v==0, self.epsilon, v)        
        dl = u * np.log(2*u/(u+v))
        dr = v * np.log(2*v/(u+v))
        return np.sum(dl + dr)

    def vicis_wave_hedges(self, u, v):
        """
        Vicis-Wave Hedges distance.

        References:
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307
        """
        with np.errstate(divide='ignore'):
            u_v = abs(u - v)
            uvmin = np.minimum(u, v)
            return np.sum(np.where(uvmin != 0, u_v/uvmin, 0))

    def vicis_symmetric_chisq(self, u, v):
        """
        Vicis Symmetric chi-square distance.

        References:
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307
        """
        with np.errstate(divide='ignore'):
            u_v = (u - v)**2
            uvmin = np.minimum(u, v)**2
            return np.sum(np.where(uvmin != 0, u_v/uvmin, 0))


    def wave_hedges(self, u, v):
        """
        Wave Hedges distance.

        References:
            1. Sung-Hyuk C (2007) Comprehensive Survey on Distance/Similarity 
               Measures between Probability Density Functions. International
               Journal of Mathematical Models and Methods in Applied Sciences.
               1(4), 300-307

        """
        with np.errstate(divide='ignore'):
            u_v = abs(u - v)
            uvmax = np.maximum(u, v)
            return np.sum(np.where(((u_v != 0) & (uvmax != 0)), u_v/uvmax, 0))
