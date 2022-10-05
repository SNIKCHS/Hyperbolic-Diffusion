Encoder: hyperboloid

-> atomic_numbers ==(b,n_atom)==

-> positions ==(b,n_atom,3)==

atomic_numbers ->embedding ==(b,n_atom,dim-4)==

h = concat [zeros==(b,n_atom,1)==,positions==(b,n_atom,3)==,embedding ==(b,n_atom,dim-4)== ]  ==(b,n_atom,dim)==

h = HNN(h) ==(b,n_atom,dim)== tangent space

