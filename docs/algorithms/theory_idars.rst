IDaRS Theory
==============

IDaRS stands for **I**\ terated **D**\ raw **a**\ nd **R**\ andom **S**\ ampling.
It was introcuced in an article that is available `here
<https://www.thelancet.com/journals/landig/article/PIIS2589-7500(2100180-1/fulltext>`_.
Pseudocode for the algorithm is given below. The algorithm infers (predicts) tile-level
labels from slide-level labels, transforming information that is originally known
only at a large scale into local information.
It is novel, general, fast and as accurate as might be hoped
for from such an algorithm.

For a tile (or tile) :math:`T`, let :math:`p(T)` denote the slide-level binary label
`False|True = 0|1` of the WSI containing :math:`T`, denoting whether the WSI
displays a certain feature :math:`F`, perhaps
a genetic abnormality (as determined by PCR or IHC, for example). So
all tiles in the same WSI have exactly the same :math:`p` label. Let :math:`q(T)`
be the 'probability' of :math:`T` having the feature :math:`F`. The loss function
is cross entropy (or some variant of it), defined by:

.. math:: L=-\sum_{T\in C}(p(T)\log(q(T)) + (1 - p(T))\log(1 -q(T)))

where :math:`C` is some set of tiles. When :math:`q` is allowed to range over all
possible :math:`q:C\to[0,1]`, this function has a unique minimum and unique
local minimum at :math:`q=p`, when its value is 0. However, here :math:`q(T)`
is constrained to depend only on the pixel values of :math:`T`, and the
cross entropy is, in practice, strictly positive.

Pseudocode for the algorithm follows:

    | ts = empty set # ts=training set  
    | for each labelled WSI W_i  
    |   create the subset S_i of tumour tiles  
    |   add r+k randomly chosen tiles of S_i to ts  
    | for each epoch  
    |   # process ts, fast because ts is small  
    |   randomize and divide ts into batches of a fixed size  
    |   for each batch  
    |       calculate loss per tile  
    |       ieverage and backpropagate the loss per batch  
    |   # prepare next training set nt  
    |   nt = empty set  
    |   for each W_i  
    |       add to nt k top probability tiles in S_i\cap ts   
    |       # alternatively use k tiles with smallest loss  
    |       add to nt r further tiles randomly chosen from S_i  
    |   ts = nt  


