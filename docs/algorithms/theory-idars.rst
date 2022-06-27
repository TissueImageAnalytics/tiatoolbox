IDaRS Theory
====================

IDaRS stands for **I**\ terative **D**\ raw **a**\ nd **R**\ ank **S**\ ampling,
an algorithm introduced in an article
[DOI: `https://doi.org/10.1016/S2589-7500(21)00180-1
<https://www.thelancet.com/journals/land$ig/article/PIIS2589-7500(2100180-1/fulltext>`_]
by **Bilal et al.**, *"Development and validation of a weakly supervised
deep learning framework to predict the status of molecular pathways
and key mutations in colorectal cancer from routine histology images:
a retrospective study"*.
*Supplementary Materials* are available.

The algorithm is used to infer (predict) tile-level
labels from slide-level labels, transforming information that is
originally known only at low resolution into high resolution information.
It is novel, general, and also faster and more accurate than
previous procedures.

In this section, we discuss the IDaRS algorithm from a theoretical
point of view. Pseudocode for the algorithm is given below.

Each WSI is divided into small rectangles, called *tiles*, all of the
same size, with
:math:`h` rows and :math:`w` columns of pixels, for some user-chosen
:math:`h` and :math:`w`.
Let :math:`F` be a particular feature of interest, which may or may not
exist in a particular tile or WSI. In the
experiments described in the paper referred to above,
:math:`F` is some genetic abnormality (perhaps determined by IHC or PCR),
but it could be any feature that makes
sense both for tiles and for WSIs, and such that, if a WSI has the
feature :math:`F`,
then some of its tiles also have that feature.
For the IDaRS procedure, it is assumed that each WSI used in this
section comes provided with a label :math:`True=1`, saying that the
WSI has the feature :math:`F`, or a label :math:`False=0` saying that
it does not.

After training, IDaRS can be used to provide the label for an
unlabelled WSI. It can also be used to locate regions with the
feature :math:`F`, potentially improving biological understanding.
To see an inference on a trained IDaRS model `[click here]
<https://github.com/TissueImageAnalytics/tiatoolbox/blob/doc-idars/examples/inference-pipelines/idars.ipynb>`_.

Let :math:`T` be a tile of one of our WSIs. We set :math:`p(T)`
equal to the label of the WSI containing :math:`T`, so that :math:`p(T)`
is equal to
either :math:`0 \text{ or } 1`, and,
as :math:`T` varies over the tiles of a fixed WSI, :math:`p(T)`
is constant.
We wish to estimate :math:`q(T)\in[0,1]`,
the probability that :math:`T` has feature :math:`F`.

Given a set :math:`\mathcal{S}` of tiles, possibly drawn from many WSIs,
we use, as the loss function,  cross
entropy (or some variant of it)
:math:`L_{\mathcal{S}}` , a real-valued function,  defined by:

.. math::

   L_{\mathcal{S}}(q) =
   \sum_{T\in \mathcal{S}}(p(T)\log(q(T))+(1-p(T))\log(1-q(T))).

This function has a unique local minimum, when its domain is the
set of ALL functions :math:`q:\mathcal{S}\to[0,1]`,
and this local minimum is also a global minimum, at :math:`q=p`.
However, we require :math:`q(T)` to depend only on the pixel values of
:math:`T`. To express this mathematically, let
:math:`\pi:\mathcal{S}\to\mathbb{R}^{h\times w\times 3}` be the map that
sends a tile :math:`T` to the RGB intensities of its :math:`h.w`  pixels.
Given a function :math:`q_0:\mathbb{R}^{h\times w\times 3}\to[0,1]`, we define
:math:`q = \pi\circ q_0`, and then :math:`L_{\mathcal{S}}(q)` can be
calculated.
We expect many local minima for :math:`L_{\mathcal{S}}` ,
each having values greater than the global minimum at :math:`q=p`.

Parameters :math:`r` and :math:`k` are chosen by the user, and the
following algorithm is applied, starting with :math:`q=p`. Successive
values for :math:`q_0`, and hence for :math:`q`,  are produced by the algorithm,
using stochastic gradient descent as usual.

    | :math:`nts = \emptyset` # Next Training Set starts empty
    | for each labelled WSI :math:`W_i`
    |   determine the subset :math:`\mathcal{S}_i` of tumour tiles
    |   add :math:`r+k` randomly chosen tiles of :math:`\mathcal{S}_i` to `nts`
    | for each epoch
    |   :math:`ts = nts`
    |   # processing :math:`ts` is fast, because :math:`ts` is comparatively small
    |   randomize and divide :math:`ts` into batches of a fixed
        convenient size
    |   for each batch
    |       calculate loss per tile
    |       # next, change the weights that determine :math:`q_0,` and, hence, also :math:`q`.
    |       backpropagate the loss per batch
    |   # create the next training set :math:`nts`
    |   :math:`nts = \emptyset`  # new training set starts empty
    |   for each :math:`W_i`
    |       to :math:`nts` add the :math:`k` top probability tiles in
            :math:`\mathcal{S}_i \cap ts`
    |       # alternatively add the :math:`k` tiles with smallest loss
    |       to :math:`nts` add :math:`r` further tiles randomly chosen
            from :math:`\mathcal{S}_i`

The above pseudocode gives a crude but correct summary of the Python
computer program discussed and explained in the paper by **Bilal
et al**, cited in the first paragraph above. The pseudocode is also
a correct summary of the slightly different IDaRS program in this
repository. It is more careful than the pseudocode presented
in the Supplement to the original paper.

The IDaRS algorithm is effective because it is very likely that the
:math:`k` chosen tiles combined with iteratively updated random :math:`r` tiles will contribute most to moving the weights in
the desired direction.
