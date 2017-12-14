(* ::Package:: *)

(* ::Code::Initialization::Bold:: *)
BeginPackage["DNAViz`"]

nwrap::usage = "Number of times the DNA superhelix wraps around the nucleosome";
\[Zeta]max::usage = "Provided for convenience, \[Zeta]max = 2\[Pi] nwrap";
dnaRadius::usage = "Radius of DNA (approx) (with units)";
superhelixRadius::usage = "Radius of DNA superhelix around nucleosome";
superhelixPitch::usage = "Pitch of DNA superhelix around nucleosome";
passiveEulerMatrix::usage = "Usage: passiveEulerMatrix[{\[Psi],\[Theta],\[Phi]}].";
anglesOfEulerMatrix::usage = "anglesOfEulerMatrix[m] extracts the angles {\[Psi],\[Theta],\[Phi]} out of a matrix m.";
axesGraphics::usage = "Graphics for axes based on unit vectors specified as rows.";
axesWithAngles::usage = "Graphics for axes based on Euler angles {\[Psi],\[Theta],\[Phi]}.";
helix::usage = "helix[\[Zeta]] gives the coordinates at angle \[Zeta] on the standard helix.";
tangentNormalOfHelix::usage = "Tangent and normal vectors on the standard helix";
tangentNormalRotMatrix::usage = "Returns {n,t\[Cross]n,t} given {t,n}.";
rotMatrixOnHelix::usage = "Convenience fn: rotation matrix on standard helix";
axesOnHelix::usage = "Convenience fn: draw axes on standard helix";
A::usage = "Resets axes aligned by Euler angles to 'standard helix' axes for standard helix
             on right multiplication.";
AInv::usage = "Transpose/Inverse of the 'A' matrix.";
arbitraryHelix::usage = "Helix at an arbitrary initial rotation";
axesOnArbitraryHelix::usage = "Axes graphics on an arbitrary helix";
nucleosomeGraphics::usage = "Draw a cylinder for a nucleosome";
nucleosomeArrayGraphic::usage =
"nucleosomeArrayGraphic[rodLength, strandAngles, nucleosomes, fakeRodAngles:{}, tubeRadius:dnaRadiusVal]
draws the graphics for a nucleosome array using rodLength and strandAngles for the rods,
successively inserting graphics for nucleosome cores (and wrapping DNA) between rods at
indices n and n+1. One can optionally visualize the fake rods inserted at the end of the
nucleosome cores (for the energy calculation) by supplying the fakeRodAngles argument.
The DNA is visualized as a tube, the radius can be altered; by default it is set to 10 \[Angstrom],
which is the correct physical value.
";

Begin["Private`"]

nwrap = 1.65;
\[Zeta]max := 2\[Pi] nwrap;
dnaRadius = Quantity[10, "Angstroms"];
dnaRadiusVal = 10;
superhelixRadius = Quantity[41.8,"Angstroms"];
r0val = 41.8;
superhelixPitch = Quantity[23.9,"Angstroms"];
z0val = 23.9;
nucleosomeRadiusFraction = 1 - dnaRadius/superhelixRadius;

{xaxis,yaxis,zaxis} = IdentityMatrix[3];
(* Mathematica uses active rotations by default but we want passive rotations *)
RotateAxes[\[Theta]_,w_]:=RotationMatrix[-\[Theta],w]
Rx[\[Theta]_]:=RotateAxes[\[Theta],xaxis]
Ry[\[Theta]_]:=RotateAxes[\[Theta],yaxis]
Rz[\[Theta]_]:=RotateAxes[\[Theta],zaxis]
passiveEulerMatrix[{\[Psi]_,\[Theta]_,\[Phi]_}]:=Rz[\[Phi]].Rx[\[Theta]].Rz[\[Psi]]

(* ArcTan[Cos[\[Alpha]],Sin[\[Alpha]]] will usually be called atan2 in other languages *)
anglesOfEulerMatrix[m_]:=
If[m[[3]][[3]]==1, {ArcTan[m[[1]][[1]],m[[1]][[2]]], 0, 0},
  If[m[[3]][[3]]==-1,
    {ArcTan[m[[1]][[1]],m[[1]][[2]]],\[Pi],0},
    Module[{sintheta=Sqrt[1-m[[3]][[3]]^2]},
      {
        ArcTan[-m[[3]][[2]]/sintheta,m[[3]][[1]]/sintheta]
      , ArcCos@m[[3]][[3]]
      , ArcTan[m[[2]][[3]]/sintheta, m[[1]][[3]]/sintheta]
      }
    ]
  ]
]

(*
Unit vectors are specified as rows. To be more specific, if we are transforming from frame S1
to S2,the vectors representing the new axes (x2,y2,z2) in the old frame S1 should be specified
as the rows.
If specifying using a passiveEulerMatrix, unitvecs should be like (using the paper's notation)
[n1 n2 t] where n1, n2 and t are columns.
The axes (x1,y1,z1) are represented by (n1,n2,t) in S' (using the paper's notation).
However, we usually want to show S2's axes moving and S1's axes fixed.
So we need the vectors for the (x2,y2,z2) in S.
This is done by taking the rows of [n1 n2 t] instead of the columns.
*)
axesGraphics[scale_,origin_,unitvecs_,mode_:"Active",opt___]:=
  Graphics3D/@(
    {{opt,Thick,Red,#1},{opt,Thick,Green,#2},{opt,Thick,Blue,#3}}&@@
      (Arrow[{origin,origin+scale #}]&/@
        (If[mode=="Active",Identity,Transpose]@unitvecs)
      )
  )
(*flipMode[mode_] := If[mode=="Passive", "Active", "Passive"];*)
axesWithAngles[scale_,origin_,angles_,mode_:"Active",opt___] :=
  axesGraphics[scale,origin,passiveEulerMatrix[angles],mode,opt]

(* left-handed helix starting at (r0,0,0) and going downwards (initially) and up *)
helix[\[Zeta]_] := {r0val Cos[\[Zeta]],-r0val Sin[\[Zeta]],z0val \[Zeta]/(2\[Pi])}
tangentNormalOfHelix[\[Zeta]_] := {
  {-r0val Sin[\[Zeta]], -r0val Cos[\[Zeta]], z0val/(2\[Pi])}/Sqrt[r0val^2+z0val^2/(4\[Pi]^2)]
  , {-Cos[\[Zeta]], Sin[\[Zeta]], 0}
}
(* For helix, we are using n and t in the usual way;
these represent "new" vectors in terms of "old" axes
so we should not take a transpose and supply [n^T;(t\[Cross]n)^T;t^T] to axesGraphics *)
tangentNormalRotMatrix[{t_,n_}] := Normalize/@{n,t\[Cross]n,t}
rotMatrixOnHelix := tangentNormalRotMatrix@*tangentNormalOfHelix
axesOnHelix[scale_,\[Zeta]_,mode_:"Active",opt___] :=
  axesGraphics[scale,helix[\[Zeta]],rotMatrixOnHelix[\[Zeta]],mode,opt]

\[Lambda] = ArcTan[#2/#3&@@tangentNormalOfHelix[0][[1]]] (* approx -1.48; slightly larger than -\[Pi]/2 *)
(* 'A' resets axes aligned by Euler angles to "standard helix" axes for starting point. *)
A[angles_] := Rx[\[Lambda]].Rz[\[Pi]].passiveEulerMatrix[angles];
AInv := Transpose @* A;

arbitraryHelix[entryR_,rot_][\[Zeta]_]:=entryR+rot.(helix[\[Zeta]]-helix[0])
axesOnArbitraryHelix[scale_,entryR_,rot_,mode_:"Active",opt___][\[Zeta]_]:=
  axesGraphics[
    scale
    , arbitraryHelix[entryR,rot][\[Zeta]]
    , rotMatrixOnHelix[\[Zeta]].Transpose[rot]
    , mode
    , opt
  ]

nucleosomeGraphics[entryR_,rot_,radiusScale_:nucleosomeRadiusFraction,opacity_:1.0]:=
  Graphics3D @ {
    Opacity[opacity]
    ,Cylinder[{#,#+rot.{0,0,nwrap*z0val}}&[entryR-rot.helix[0]], radiusScale*r0val]
  }
End[]

splitList[xs_,inds_]:=
xs[[#1;;#2]]&@@@
Partition[Append[Prepend[1]@Riffle[inds,inds+1],Length@xs],2];
(* splitList[Range[10],{3,7}]\[Equal]{{1,2,3},{4,5,6,7},{8,9,10}} *)

tangentVector[{\[Phi]_,\[Theta]_,\[Psi]_}]:={Sin[\[Theta]] Sin[\[Phi]],Cos[\[Phi]] Sin[\[Theta]],Cos[\[Theta]]};
normalVector[{\[Phi]_,\[Theta]_,\[Psi]_}]:={
    Cos[\[Phi]] Cos[\[Psi]] - Cos[\[Theta]] Sin[\[Phi]] Sin[\[Psi]]
    , - Cos[\[Psi]] Sin[\[Phi]] - Cos[\[Theta]] Cos[\[Phi]] Sin[\[Psi]]
    , Sin[\[Theta]] Sin[\[Psi]]
  };

nucleosomeEnd[a_,tscale_,dummyRod_:{},tubeRadius_:dnaRadiusVal] :=
  Module[{R0=a[["startCoord"]], t=a[["prevTangent"]], n=a[["prevNormal"]]
      , rot, helixEndCoordinate, dummyRodGraphic
    },
      rot = AInv@*anglesOfEulerMatrix@tangentNormalRotMatrix[{t, n}];
      helixEndCoordinate = arbitraryHelix[R0, rot][\[Zeta]max];
      dummyRodGraphic := Graphics3D@{
          Red, Dashed, Thickness[Large], Arrowheads[0.01]
          , Arrow[{helixEndCoordinate, helixEndCoordinate + tscale*tangentVector[dummyRod]}]
        };
    <|
      "helixEndCoordinate" -> helixEndCoordinate
      , "graphics" -> If[dummyRod=={}, Identity, Prepend[dummyRodGraphic]]@{
          nucleosomeGraphics[R0, rot]
          , Graphics3D@{Tube[arbitraryHelix[R0, rot]/@Range[0, \[Zeta]max, 0.1], tubeRadius]}
        }
    |>
  ];

nucleosomeArrayGraphic[rodLength_, strandAngles_, endAngle_, nucleosomes_
                            , fakeRodAngles_:{}, tubeRadius_:dnaRadiusVal]:=
Module[{
  tscale, nscale, tangents, normals, tmpFakeRodAngles
  , data, coords
  , sowCoords, sowGraphics, linkerSegment
  , arrows, tubes
  , normalArrows, endArrow, mainStrand
},

  tscale = 10 rodLength; (* 10 because drawing is in \[Angstrom] whereas given values are in nm *)
  nscale = 1.5 tscale;
  tangents = tscale splitList[tangentVector/@strandAngles, nucleosomes];
  normals= nscale splitList[normalVector/@strandAngles,nucleosomes];
  tmpFakeRodAngles = If[fakeRodAngles=={},
                         Table[{}, Length[nucleosomes]+1],
                         Prepend[{}]@fakeRodAngles];
  (*
    Saves the coordinates for a strand between consecutive nucleosomes for drawing and
    returns the final value, which is the starting coordinate for the next superhelix.
  *)
  sowCoords[strandCoords_] := (Sow[strandCoords,"c"]; strandCoords[[-1]]);
  sowGraphics[superHelix_] := (Sow[superHelix[["graphics"]],"g"]; superHelix[["helixEndCoordinate"]]);
  linkerSegment[acc_,{strandTangents_,strandNormals_,dummyRod_}] :=
    (* We should draw a nucleosome if and only if startCoord doesn't match the attachment base. *)
    Module[{strandStart = If[acc[["startCoord"]]=={0,0,0}, {0,0,0},
                              sowGraphics@nucleosomeEnd[acc,tscale,dummyRod,tubeRadius]]},
      <|
        "startCoord"-> sowCoords@*Accumulate@*Prepend[strandStart]@strandTangents
        , "prevTangent"->strandTangents[[-1]], "prevNormal"->strandNormals[[-1]]
      |>
    ];
  data = #1->{##2}&@@@
    Reap[
      (* First save the keys for making the rules later. *)
      Sow["graphics","g"];
      Sow["coords","c"];
      (* Each operation of the fold builds segments of linker DNA *)
      FoldList[linkerSegment, <|"startCoord"->{0,0,0}|>
               , Transpose[{tangents,normals,tmpFakeRodAngles}]]
   ][[2]];
  tubes[xs_] := Graphics3D@{Tube[#,tubeRadius]&/@xs};
  arrows[xs_, color_:Black] :=
    Graphics3D@{Thickness[Large], Arrowheads[0.01], color, (Arrow(*@*Tube*))/@xs};
  (* TODO: Explain why the last element is excluded. *)
  normalArrows = arrows@*Transpose/@MapThread[{#1[[;;-2]],#1[[;;-2]]+#2}&,{"coords"/.data,normals}];
  (* The last coordinate of the last linker is attached to the bead. *)
  endArrow = arrows[{{#,#+nscale*normalVector[endAngle]}}&[("coords"/.data)[[-1, -1]]]
      , Blue];
  (* last coordinate should be excluded with \[LeftDoubleBracket];;-2\[RightDoubleBracket] as there are k normals for k+1 rods *)
  mainStrand = tubes["coords"/.data];

  Flatten[{"graphics"/.data, normalArrows, mainStrand, endArrow}]
]

EndPackage[]
