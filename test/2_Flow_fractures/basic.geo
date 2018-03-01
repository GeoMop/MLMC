cl1 = 0.05;
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


Point(5) ={0.02,0.45,0,cl2};
Point(6) ={0.98,0.87,0,cl2};
Point(7) ={0.25,0.07,0,cl2};
Point(8) ={0.23,0.17,0,cl2};
Point(9) ={0.38,0.63,0,cl2};
Point(10) ={0.36,0.73,0,cl2};
Point(11) ={0.69,0.84,0,cl2};
Point(12) ={0.68,0.94,0,cl2};
Point(13) ={0.81,0.92,0,cl2};
Point(14) ={0.81,0.98,0,cl2};
Point(15) ={0.21,0.42,0,cl2};
Point(16) ={0.06,0.87,0,cl2};

Line(20) ={5,6};
Line(21) ={7,8};
Line(22) ={9,10};
Line(23) ={11,12};
Line(24) ={13,14};
Line(25) ={15,16};

Physical Line("frac_1") = {20};
Physical Line("frac_2") = {21};
Physical Line("frac_3") = {22};
Physical Line("frac_4") = {23};
Physical Line("frac_5") = {24};
Physical Line("frac_6") = {25};

Line{20:25} In Surface{32};