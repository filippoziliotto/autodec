1) Does it make sense to make SQ loss based on symmetry? Cause i see that for many objects you should have the same superquadric just shifted so you could in theory reduce the number of superquadric by adding a symmetry constraint. Can this constraint be learned, trough a specific loss?
Main reason, i see that many objects should have basically the same superquadric just "shifted/mirrored", yet they have different SQ values, like a lef and right parts of a plane wing.

2) Editable scenes by reconstructing the e.g. shifted SQ?

3) Can we have a constraint for "smooth" SQ? so that they "blend" one with nother smoothly and better represent the object (also visually)? e.g. a Boundary proximity loss, Decoder continuity loss.
Main reason many SQ i see represent correctly the object but could be more "blent" together and visually more representative.