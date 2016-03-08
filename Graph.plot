set term png transparent enhanced size 1280, 720 
#set terminal png size 1280, 720 
set boxwidth
set key on
set style line 11 lc rgb '#808080' lt 1
set ylabel "Episodes" font "Arial-Bold, 18"
set xlabel "Time Steps" font "Arial-Bold, 18"
set style function lines
set title font "Courier New, 20"



set label 1 "{/Symbol a}=0.5" at 100,180 font "Arial-Bold, 20" textcolor lt 1 
set label 2 "{/Symbol g}=1.0" at 100,170 font "Arial-Bold, 20" textcolor lt 2

set label 3 "{/Symbol e}=0.1" at 100,160 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA(0),{/Symbol e}-Soft" at 100,190 font "Arial-Bold, 20" textcolor lt 4
set title "Non-Stochastic Wind with Standard Moves"
set output "Output/Figure1.png"
plot "Output/V1.data" using 1:2 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/Figure2.png"
plot "Output/V2.data" using 1:2 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/Figure3.png"
plot "Output/V3.data" using 1:2 with lines lw 4 title column  


#set label 3 "{/Symbol e}=0.1" at 100,160 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA(0),UCB" at 100,190 font "Arial-Bold, 20" textcolor lt 4

set title "Non-Stochastic Wind with Standard Moves"
set output "Output/Figure4.png"
plot "Output/V4.data" using 1:2 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/Figure5.png"
plot "Output/V5.data" using 1:2 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/Figure6.png"
plot "Output/V6.data" using 1:2 with lines lw 4 title column


set label 3 "{/Symbol e}=0.1" at 100,160 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA({/Symbol l}),{/Symbol e}-Soft" at 100,190 font "Arial-Bold, 20" textcolor lt 4
set label 5 "{/Symbol l}=0.1" at 100,151 font "Arial-Bold, 20" textcolor lt 3 

set title "Non-Stochastic Wind with Standard Moves"
set output "Output/Figure7.png"
plot "Output/V7.data" using 1:2 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/Figure8.png"
plot "Output/V8.data" using 1:2 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/Figure9.png"
plot "Output/V9.data" using 1:2 with lines lw 4 title column 


#set label 3 "{/Symbol e}=0.1" at 100,160 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA({/Symbol l}),UCB" at 100,190 font "Arial-Bold, 20" textcolor lt 4

set title "Non-Stochastic Wind with Standard Moves"
set output "Output/Figure10.png"
plot "Output/V10.data" using 1:2 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/Figure11.png"
plot "Output/V11.data" using 1:2 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/Figure12.png"
plot "Output/V12.data" using 1:2 with lines lw 4 title column



set label 5 "{/Symbol a}=0.5" at 1000,3600 font "Arial-Bold, 20" textcolor lt 1 
set ylabel "Steps in each Episode" font "Arial-Bold, 18"
set xlabel "Episodes" font "Arial-Bold, 18"
set yrange [0:400]

set label 1 "{/Symbol a}=0.5" at 100,360 font "Arial-Bold, 20" textcolor lt 1 
set label 2 "{/Symbol g}=1.0" at 100,340 font "Arial-Bold, 20" textcolor lt 2

set label 3 "{/Symbol e}=0.1" at 100,320 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA(0),{/Symbol e}-Soft" at 100,380 font "Arial-Bold, 20" textcolor lt 4
set title "Non-Stochastic Wind with Standard Moves"
set output "Output/FigureX1.png"
plot "Output/V1.data" using 2:3 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/FigureX2.png"
plot "Output/V2.data" using 2:3 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/FigureX3.png"
plot "Output/V3.data" using 2:3 with lines lw 4 title column  


#set label 3 "{/Symbol e}=0.1" at 100,320 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA(0),UCB" at 100,380 font "Arial-Bold, 20" textcolor lt 4

set title "Non-Stochastic Wind with Standard Moves"
set output "Output/FigureX4.png"
plot "Output/V4.data" using 2:3 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/FigureX5.png"
plot "Output/V5.data" using 2:3 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/FigureX6.png"
plot "Output/V6.data" using 2:3 with lines lw 4 title column


set label 3 "{/Symbol e}=0.1" at 100,320 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA({/Symbol l}),{/Symbol e}-Soft" at 100,380 font "Arial-Bold, 20" textcolor lt 4
set label 5 "{/Symbol l}=0.1" at 100,300 font "Arial-Bold, 20" textcolor lt 3 

set title "Non-Stochastic Wind with Standard Moves"
set output "Output/FigureX7.png"
plot "Output/V7.data" using 2:3 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/FigureX8.png"
plot "Output/V8.data" using 2:3 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/FigureX9.png"
plot "Output/V9.data" using 2:3 with lines lw 4 title column 


#set label 3 "{/Symbol e}=0.1" at 100,320 font "Arial-Bold, 20" textcolor lt 3 
set label 4 "SARSA({/Symbol l}),UCB" at 100,380 font "Arial-Bold, 20" textcolor lt 4

set title "Non-Stochastic Wind with Standard Moves"
set output "Output/FigureX10.png"
plot "Output/V10.data" using 2:3 with lines lw 4 title column 

set title "Non-Stochastic Wind with King's Moves" font "Courier New, 20"
set output "Output/FigureX11.png"
plot "Output/V11.data" using 2:3 with lines lw 4 title column  

set title "Stochastic Wind with Kings Moves" font "Courier New, 20"
set output "Output/FigureX12.png"
plot "Output/V12.data" using 2:3 with lines lw 4 title column