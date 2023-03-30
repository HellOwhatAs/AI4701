#!/usr/bin/env wolframscript


SetDirectory[NotebookDirectory[]];
img = Import["./lines.png"];
table = Grid[
   Prepend[
      Table[
         {
            N[i/15],
            Length[lines = ImageLines[img, i/15, Method -> "Hough"]],
            HighlightImage[img, lines, ImageSize -> Medium]
         },
         {i, 2, 6}
      ],
      {"threshold", "line count", "result"}
   ], Frame -> All]
Export["./Hough.svg", table];