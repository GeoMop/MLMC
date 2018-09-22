// Gmsh project created on Tue May 29 16:51:29 2018

Mesh.Algorithm=5;

//+
SetFactory("OpenCASCADE");
//+
SetFactory("Built-in");
//+
Point(1) = {1, 0, -1, 1.0};
//+
Point(2) = {-1, 0, -1, 1.0};
//+
Point(3) = {-1, 0, 0.5, 1.0};
//+
Point(4) = {1, 0, 0.2, 1.0};
//+
Point(5) = {-0, 0, -0.4, 1.0};
//+
Point(6) = {-0, 0, -0.5, 1.0};
//+
Point(7) = {0.3, 0, -0.5, 1.0};
//+
Point(8) = {0.3, 0, -0.4, 1.0};
//+
Point(9) = {1.6, 0, 0.2, 1.0};
//+
Recursive Delete {
  Point{9}; 
}
//+
Point(9) = {0.7, 0, 0.1, 1.0};
//+
Point(10) = {0.4, 0, 0.1, 1.0};
//+
Point(11) = {-0, 0, 0.3, 1.0};
//+
Point(12) = {-0.3, 0, 0.4, 1.0};
//+
Point(13) = {-0.8, 0, 0.7, 1.0};
//+
Point(14) = {-0.5, 0, 0.7, 1.0};
//+
Point(15) = {-0.2, 0, 0.6, 1.0};
//+
Spline(1) = {4, 9, 10, 11, 15, 14, 3};
//+
Point(16) = {1, 0, -0.1, 1.0};
//+
Point(17) = {0.5, 0, -0.2, 1.0};
//+
Point(18) = {-0, 0, -0.1, 1.0};
//+
Point(19) = {-0.5, 0, 0, 1.0};
//+
Point(20) = {-1, 0, 0.1, 1.0};
//+
Point(21) = {-0.5, 0, 0.2, 1.0};
//+
Spline(2) = {16, 17, 18, 21, 20};
//+
Line(3) = {4, 16};
//+
Line(4) = {1, 16};
//+
Line(5) = {1, 2};
//+
Line(6) = {2, 20};
//+
Line(7) = {20, 3};
//+
Line(8) = {5, 8};
//+
Line(9) = {8, 7};
//+
Line(10) = {7, 6};
//+
Line(11) = {6, 5};
//+
Recursive Delete {
  Point{12}; 
}
//+
Recursive Delete {
  Point{19}; 
}
//+
Recursive Delete {
  Point{13}; 
}
//+
Translate {-0.3, 0, 0} {
  Point{8}; Line{8}; Point{5}; Line{11}; Point{6}; Line{10}; Point{7}; Line{9}; 
}
//+
Translate {0.1, 0, 0} {
  Point{11}; 
}
//+
Line Loop(1) = {1, -7, -2, -3};
//+
Plane Surface(1) = {1};
//+
Line Loop(2) = {2, -6, -5, 4};
//+
Line Loop(3) = {8, 9, 10, 11};
//+
Plane Surface(2) = {2, 3};
//+
Plane Surface(3) = {3};
//+
Physical Line(".surface") = {1};
//+
Physical Surface("ground_0") = {1};
//+
Physical Surface("ground_1") = {2};
//+
Physical Surface("repo") = {3};
//+
Field[1] = Max;
//+
Field[1].FieldsList = {0, 1};
//+
Field[2] = Param;
//+
Field[2].FX = "0.1";
//+
Field[2].FY = "0.1";
//+
Field[2].FZ = "0.1";
//+
Field[3] = MathEval;
//+
Delete Field [1];
//+
Delete Field [2];
//+
Field[3].F = "0.1";
