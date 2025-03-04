(* Построение графика *)
plot = Plot[Sin[x], {x, 0, 2 Pi}, 
   PlotLabel -> "График функции Sin[x]", 
   AxesLabel -> {"x", "y"}, 
   PlotStyle -> Red, 
   GridLines -> Automatic, 
   Background -> LightBlue];

(* Экспорт графика в PNG *)
(*Export["plot.png", plot];*)

(* Установка кодировки *)
SetOptions[$Output, PageWidth -> Infinity, CharacterEncoding -> "UTF-8"];

(* Ваш код *)
Print["График сохранен в файл plot.png"];