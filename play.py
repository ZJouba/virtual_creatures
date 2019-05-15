
letters = "abcd"
charlist = [char for char in letters]

print(np.random.choice(charlist))


F+[[]-]-F[-F]+

'FF+[[F+[[]-]-F[-F]+]-F+[[]-]-F[-F]+]-FF[-FFF+[[]-]-F[-F]+]+F+[[]-]-F[-F]+~'

2,4  - 45 |

0,2  - 45 | 0,2  - 45 |1,3  - 0 |


0,0 | 0, 1``````| 0,2 | 1,3 | 2,4 | 2,5 |

Character        Meaning
   F	         Move forward by line length drawing a line
   f	         Move forward by line length without drawing a line
   +	         Turn left by turning angle
   -	         Turn right by turning angle
   |	         Reverse direction (ie: turn by 180 degrees)
   [	         Push current drawing state onto stack
   ]	         Pop current drawing state from the stack
   #	         Increment the line width by line width increment
   !	         Decrement the line width by line width increment
   @	         Draw a dot with line width radius
   {	         Open a polygon
   }	         Close a polygon and fill it with fill colour
   >	         Multiply the line length by the line length scale factor
   <	         Divide the line length by the line length scale factor
   &	         Swap the meaning of + and -
   (	         Decrement turning angle by turning angle increment
   )	         Increment turning angle by turning angle increment

self.angle = angle
        self.point = point
        self.vector = vector
        self.length = length
        self.turning_angle_inc = turning_angle_inc
        self.lenght_scale_factor = lenght_scale_factor