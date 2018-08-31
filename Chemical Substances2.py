"""
The UN Chemical Weapons Inspectors investigate a warehouse storing some chemical
substances for manufacturing deadly chemical weapons. There are n2 chemical containers that are arranged in an nxn array. The UN Chemical Weapons Inspectors look into each chemical container in the warehouse and store the information into a 2-dimensional array; ‘0’ for an empty container, and ‘k’ for a container holding the chemical substance type k, where k = 1, 2, 3, …, 9. The following figure shows an example of the 9x9 array obtained from a warehouse. 

 

 
The UN Chemical Weapons Inspectors found the following three facts from the chemical substances in the warehouse. 

1. The containers holding chemical substances form a rectangular shape. There is no empty container in the rectangle. For example, there are three rectangles of the containers holding chemical substances—A, B, and C—in the above figure. 

2. The dimension (the number of rows x the number of columns) of a rectangle of the containers holding chemical substances is not the same as that of any other rectangle. For example, A is 3x4, B is 2x3, and C is 4x5 in the above figure.

3. There are empty containers between two rectangles of the containers holding chemical substances. For example, you can find the empty containers (‘0’ elements) between A and B as well as B and C. Note that there may not be an empty container between two rectangles that are adjacent diagonally; for example, there is no empty container between A and C in the figure.
 
 
The UN Chemical Weapons Inspectors have also found that there are specific relationships among the rectangles of the containers holding chemical substances through further investigations. The relationships are found that a rectangle of the containers holding chemical substances is a matrix and that the chemical substances in the containers are mixed as if we multiply two matrices. That is, multiplying two elements in the matrices is mixing two chemical substances corresponding to the two elements. Note that we ignore the time for mixing two chemical substances. 
 
 
In the above figure, since there are three matrices, A, B, and C whose dimensions are 3x4, 2x3, and 4x5, respectively, the sequence of multiplications among these matrices should be BxAxC. However, depending on which pair of matrices are multiplied first, we get a different number of the number of multiplications of elements. For example, for the above three matrices (A(3x4), B(2x3), C(4x5)), if we perform (B*A)*C—that is, B*A is done first, and then the result of (B*A) is multiplied—then there are 64 multiplications of elements. But B*(A*C) requires 90 multiplications. 
 
  
To save time for the UN investigations, write a program to find matrices (rectangles of chemical containers) from a given 2-dimensional array and to find the smallest number of multiplications of elements among the matrices found. 
 
 
[Constrains]
n is not greater than 100.
The number of rows of a matrix is different each other and so is the number of columns of a matrix. For example, when A(3x4), B(2x3), and C(4x5) are found as in the figure, their numbers of rows—3, 2, 4—are all different and similarly their numbers of columns—4, 3, 5— are all different each other. Therefore, there exists only one sequence of matrices multiplications for a given test case. 
There is no test case that cannot multiply the matrices due to dimension mismatches.


 
 
[Input]

In the first line, the number of test cases is given. Each test case consists of (n+1) lines; the first line has a positive integer n and the next n lines have an nxn matrix row-by-row, one row per line.
 
 

 
[Output]

Print out the size of the smallest number of multiplications of elements for multiplying the sequence of matrices extracted from the given test case, starting with ‘#x’, where x is the case number.

Please refer to the following information of test cases.

Group 1. n <= 10 and the number of sub matrix <= 5

Group 2. n <= 40 and 5 < sub matrix <= 10

Group 3. 40 < n <= 80 and 5 < sub matrix <= 10

Group 4. 40 < n <= 80 and 10 < sub matrix <= 15

Group 5. 80 < n <=100 and 15 < sub matrix <= 20 
 
 
[Sample Input and Output]
- Input
10 <- the number of test cases
9  <- test case #1
1 1 3 2 0 0 0 0 0 
3 2 5 2 0 0 0 0 0
2 3 3 1 0 0 0 0 0
0 0 0 0 4 5 5 3 1 <- 9x9 matrix in row-by-row
1 2 5 0 3 6 4 2 1
2 3 6 0 2 1 1 4 2
0 0 0 0 4 2 3 1 1
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
4 <- test case #2
1 2 0 0
0 0 0 0
9 4 2 0
1 7 3 0
…

- Output
#1 64
#2 6
…

"""