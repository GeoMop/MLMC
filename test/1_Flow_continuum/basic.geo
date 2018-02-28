cl1 = 0.1; 
cl2 = 0.025; 
Point(1) = {0, 0, 0, cl1};
Point(2) = {1, 0, 0, cl1};
Point(3) = {1, 1, 0, cl1};
Point(4) = {0, 1, 0, cl1};
Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {3, 4};
Line(14) = {4, 1};

Line Loop(31) = {11,12,13,14};
Plane Surface(32) = {31};

Physical Line(".bc_south") = {11};
Physical Line(".bc_east") = {12};
Physical Line(".bc_north") = {13};
Physical Line(".bc_west") = {14};

Physical Surface("matrix") = {32};

