IDaRS Theory
==============

IDaRS stands for **I**\ terated **D**\ raw **a**\ nd **R**\ andom **S**\ ampling, an
algorithm introcuced in an article available by `clicking here
<https://www.thelancet.com/journals/landig/article/PIIS2589-7500(2100180-1/fulltext>`_.
Pseudocode for the algorithm is given below. The algorithm is used to infer (predict)
tile-level
labels from slide-level labels, transforming information that is originally known
only at a coarse scale into local information.
It is novel, general, and also faster and more accurate than competitive algorithms
at the time of writing (May 2022).

Each WSI is divided into small rectangles, called *tiles*, all of the same size, with
:math:`r` rows of pixels and :math:`c` columns, for some fixed :math:`r` and
:math:`c`.
Let :math:`F` be a particular feature of interest, which may or may not exist in a
particular tile or WSI. In the
experiments described in the paper referred to above,
:math:`F` is some genetic abnormality (perhaps determined by IHC or PCR),
but it could be any feature that makes
sense both for tiles and for WSIs, and such that, if a WSI has the feature :math:`F`,
then some of its tiles also have that feature.
For the IDaRS procedure, it is assumed that each WSI used in this section comes
provided with a label 
saying that the WSI either has (label=1)  or does not have (label=0) the feature.

Once trained IDaRS can be used to provide the label for an unlabelled WSI. It
can also be used to locate regions with the feature :math:`F`, which could
assist biological understanding

Let :math:`T` be a tile of one of our WSIs. We set :math:`p(T)`
equal to the label of the WSI containing :math:`T`, so that :math:`p(T)` is equal to
either 0 or 1, and, as :math:`T` varies over the tiles of a fixed WSI, :math:`p(T)`
is constant.
We wish to estimate :math:`q(T)\in[0,1]`,
the probability that :math:`T` has feature :math:`F`.

Given a set :math:`\mathcal{T}` of tiles, passibly drawn from many WSIs, we use cross
entropy (or some variant of it)

.. math:: L_{\mathcal{T}}(q) = -\sum_{T\in \mathcal{T}}(p(T)\log(q(T))+(1-p(T))\log(1-q(T)))

as the loss function. This function has a unique local minimum, which is also a global
minimum, namely when :math:`a(T) == p(T)`, for each tile :math:`T\in\mathcal{T}`,
when :math:`L_{q,\mathcal{T}} == 0`.
However, we require :math:`q(T)` to depend only on the pixel values :math:`T`, so that
we are really looking for a function
:math:`q:\mathbb{R}^{r\times c\times 3}\to \mathbb{R}`. Moreover, we require :math:`qz
to be continuous, so that indistinguishly small variations in pixel intensity make
very little difference to the loss. In this case, the (local) minima for
:math:`L_{q,\mathcal{T}}` are, in practice, positive.

We start the IDaRS process with :math:`q(T)=p(T)` for each tile :math:`T` and then
use stochastic gradient descent, as usual in Deep Learning,
to minimize (at least locally) the function :math:`L_{\mathcal{T}}`.

Pseudocode for the algorithm follows:

    | :math:`ts = \emptyset` # training set starts empty
    | for each labelled WSI :math:`W_i`  
    |   create the subset :math:`S_i` of tumour tiles  
    |   add :math:`r+k` randomly chosen tiles of :math:`S_i` to `ts`  
    | for each epoch  
    |   # process :math:`ts', fast because :math:`ts` is small  
    |   randomize and divide :math:`ts` into batches of a fixed convenient size  
    |   for each batch  
    |       calculate loss per tile  
    |       backpropagate the loss per batch  
    |   # creaie next training set nt  
    |   :math:`nt = \emptyset`  # new training set starts empty
    |   for each :math:`W_i`  
    |       add to :math:`nt`  :math:`k` top probability tiles in :math:`S_i`\cap ts`   
    |       # alternatively use :math:`k` tiles with smallest loss  
    |       add to :math:`nt` :math:`r` further tiles randomly chosen from :math:`S_i`  
    |   :math:`ts = nt`  
