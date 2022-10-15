Introduction to Statistical Learning.
================

Statistical Learning refers to a vast set of tools for understanding
data. These tools can be classified as *supervised* and *unsupervised*.
Broadly speaking, *supervised statistical learning* involves building a
statistical model for predicting, or estimating, an output based on the
labeled datasets. With *unsupervised statistical learning*, there are
inputs but no supervising outputs, or building a statistical model for
prediction based on the unlabeled datasets.

## Notation

For the most part of these notebooks I adopt the following notation:

I will use *n* to represent the number of distinct data points, or
observations, in our sample. I will let *p* denote the number of
variables that are available for making predictions. I will let
![x\_{ij}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_%7Bij%7D "x_{ij}")
represent the value of the
![j^{th}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;j%5E%7Bth%7D "j^{th}")
variable for the
![i^{th}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%5E%7Bth%7D "i^{th}")
observation, where
![i = 1,2,...,n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%20%3D%201%2C2%2C...%2Cn "i = 1,2,...,n")
and
![j = 1,2,...,n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;j%20%3D%201%2C2%2C...%2Cn "j = 1,2,...,n").
Throughout these notebooks and future notebooks, *i* will be used to
index the samples or observations (from 1 to *n*) and *j* will be used
to index the variables (from 1 to *p*). I let
![\\boldsymbol{X}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D "\boldsymbol{X}")
denote
![n \\times p](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n%20%5Ctimes%20p "n \times p")
matrix, where *n* represents the rows or the observations or the data
points and p represents the columns or the variables.

![ \\boldsymbol{X} = \\begin{bmatrix}
x\_{11} & x\_{12} & \\dots & x\_{1p} \\\\
x\_{21} & x\_{22} & \\dots & x\_{2p} \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
x\_{n1} & x\_{n2} & \\dots & x\_{np} \\\\
\\end{bmatrix}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%20%5Cboldsymbol%7BX%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%0Ax_%7B11%7D%20%26%20x_%7B12%7D%20%26%20%5Cdots%20%26%20x_%7B1p%7D%20%5C%5C%0Ax_%7B21%7D%20%26%20x_%7B22%7D%20%26%20%5Cdots%20%26%20x_%7B2p%7D%20%5C%5C%0A%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%0Ax_%7Bn1%7D%20%26%20x_%7Bn2%7D%20%26%20%5Cdots%20%26%20x_%7Bnp%7D%20%5C%5C%0A%5Cend%7Bbmatrix%7D " \boldsymbol{X} = \begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1p} \\
x_{21} & x_{22} & \dots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \dots & x_{np} \\
\end{bmatrix}")

At times we will be interested in the rows of
![\\boldsymbol{X}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D "\boldsymbol{X}"),
which I write as vector
![x_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i "x_i")
of length *p*, where i indicates the row number or the entry number:

![x_i = \\begin{pmatrix} 
x\_{i1} \\\\ x\_{i2} \\\\ \\vdots \\\\ x\_{ip}
\\end{pmatrix}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i%20%3D%20%5Cbegin%7Bpmatrix%7D%20%0Ax_%7Bi1%7D%20%5C%5C%20x_%7Bi2%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20x_%7Bip%7D%0A%5Cend%7Bpmatrix%7D "x_i = \begin{pmatrix} 
x_{i1} \\ x_{i2} \\ \vdots \\ x_{ip}
\end{pmatrix}")

At other times we will instead be interested in the columns of
![\\boldsymbol{X}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D "\boldsymbol{X}"),
which I write as
![\\textbf{x}\_1,\\textbf{x}\_2, \\dots, \\textbf{x}\_p](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7Bx%7D_1%2C%5Ctextbf%7Bx%7D_2%2C%20%5Cdots%2C%20%5Ctextbf%7Bx%7D_p "\textbf{x}_1,\textbf{x}_2, \dots, \textbf{x}_p"),
where p indicates the variable or the column number. Each is a vector of
length *n*. That is,

![\\textbf{x}\_j = \\begin{pmatrix} 
x\_{1j} \\\\ x\_{2j} \\\\ \\vdots \\\\ x\_{nj}
\\end{pmatrix}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7Bx%7D_j%20%3D%20%5Cbegin%7Bpmatrix%7D%20%0Ax_%7B1j%7D%20%5C%5C%20x_%7B2j%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20x_%7Bnj%7D%0A%5Cend%7Bpmatrix%7D "\textbf{x}_j = \begin{pmatrix} 
x_{1j} \\ x_{2j} \\ \vdots \\ x_{nj}
\end{pmatrix}")

So our dataset or the matrix
![\\boldsymbol{X}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D "\boldsymbol{X}")
can be represented as:

![\\boldsymbol{X} = 
\\begin{pmatrix}
\\textbf{x}\_1 & \\textbf{x}\_2 & \\dots & \\textbf{x}\_p 
\\end{pmatrix}, 
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D%20%3D%20%0A%5Cbegin%7Bpmatrix%7D%0A%5Ctextbf%7Bx%7D_1%20%26%20%5Ctextbf%7Bx%7D_2%20%26%20%5Cdots%20%26%20%5Ctextbf%7Bx%7D_p%20%0A%5Cend%7Bpmatrix%7D%2C%20%0A "\boldsymbol{X} = 
\begin{pmatrix}
\textbf{x}_1 & \textbf{x}_2 & \dots & \textbf{x}_p 
\end{pmatrix}, 
")

or

![
\\boldsymbol{X} = \\begin{pmatrix}
x^T_1 \\\\ x^T_2 \\\\ \\vdots \\\\ x^T_n 
\\end{pmatrix}
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%0A%5Cboldsymbol%7BX%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%0Ax%5ET_1%20%5C%5C%20x%5ET_2%20%5C%5C%20%5Cvdots%20%5C%5C%20x%5ET_n%20%0A%5Cend%7Bpmatrix%7D%0A "
\boldsymbol{X} = \begin{pmatrix}
x^T_1 \\ x^T_2 \\ \vdots \\ x^T_n 
\end{pmatrix}
")

I use
![y_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;y_i "y_i")
to denote the
![i^{th}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%5E%7Bth%7D "i^{th}")
observation of the variable on which we wish to make predictions, also
known as the true label. It’ll be a vector of length same as the number
of observations or the data points, *n*:

![\\textbf{y} = \\begin{pmatrix}
y_1 \\\\ y_2 \\\\ \\vdots \\\\ y_n
\\end{pmatrix}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7By%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%0Ay_1%20%5C%5C%20y_2%20%5C%5C%20%5Cvdots%20%5C%5C%20y_n%0A%5Cend%7Bpmatrix%7D "\textbf{y} = \begin{pmatrix}
y_1 \\ y_2 \\ \vdots \\ y_n
\end{pmatrix}")

Then our observed data consists of
![\\{(x_1, y_1), (x2, y2), \\dots, (x_n, y_n)\\}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5C%7B%28x_1%2C%20y_1%29%2C%20%28x2%2C%20y2%29%2C%20%5Cdots%2C%20%28x_n%2C%20y_n%29%5C%7D "\{(x_1, y_1), (x2, y2), \dots, (x_n, y_n)\}"),
where
![x_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i "x_i")
is the vector of length p. (If p = 1, then
![x_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i "x_i")
is simply a scalar.)

Matrices will be denoted using *bold capitals*, such as
![\\textbf{A}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7BA%7D "\textbf{A}").
Random variables will be denoted using *capital normal format*,
e.g. ![\\text{A}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctext%7BA%7D "\text{A}"),
regardless of their dimensions. To indicate that an object is a scalar,
we will use
![\\text{a}\\in\\mathbb{R}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctext%7Ba%7D%5Cin%5Cmathbb%7BR%7D "\text{a}\in\mathbb{R}").
To indicate that it is a vector of length
![\\text{k}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctext%7Bk%7D "\text{k}"),
I will use
![\\text{a}\\in\\mathbb{R}^k](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctext%7Ba%7D%5Cin%5Cmathbb%7BR%7D%5Ek "\text{a}\in\mathbb{R}^k")
(or
![\\text{a}\\in\\mathbb{R}^n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctext%7Ba%7D%5Cin%5Cmathbb%7BR%7D%5En "\text{a}\in\mathbb{R}^n")
if it is of length
![\\text{n}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctext%7Bn%7D "\text{n}")).
I will indicate that an object is an
![r \\times s](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;r%20%5Ctimes%20s "r \times s")
matrix using
![\\textbf{A}\\in\\mathbb{R}^{r\\times s}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7BA%7D%5Cin%5Cmathbb%7BR%7D%5E%7Br%5Ctimes%20s%7D "\textbf{A}\in\mathbb{R}^{r\times s}")

To further grasp the notation, let’s look at an example. Now let’s
pretend we have access to every patients’ medical history in a hospital,
which looks something this:

    ##      Age  Glucose Lvl Diabetic
    ## PId                           
    ## P1    12          110      Yes
    ## P2    41          130       No
    ## P3    31          145       No
    ## P4    45          111      Yes
    ## P5    56          123       No
    ## P6    65          321      Yes

Let’s give the above notation a go with this simple and very random set
of numbers. Let’s write matrix
![\\boldsymbol{X}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D "\boldsymbol{X}"):

![ \\boldsymbol{X} = \\begin{bmatrix}
P1 & 12 & 110 & Yes\\\\
P2 & 41 & 130 & No\\\\
P3 & 31 & 145 & No\\\\
P4 & 45 & 111 & Yes\\\\
P5 & 56 & 123 & No\\\\
P6 & 65 & 321 & Yes
\\end{bmatrix}
](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%20%5Cboldsymbol%7BX%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%0AP1%20%26%2012%20%26%20110%20%26%20Yes%5C%5C%0AP2%20%26%2041%20%26%20130%20%26%20No%5C%5C%0AP3%20%26%2031%20%26%20145%20%26%20No%5C%5C%0AP4%20%26%2045%20%26%20111%20%26%20Yes%5C%5C%0AP5%20%26%2056%20%26%20123%20%26%20No%5C%5C%0AP6%20%26%2065%20%26%20321%20%26%20Yes%0A%5Cend%7Bbmatrix%7D%0A " \boldsymbol{X} = \begin{bmatrix}
P1 & 12 & 110 & Yes\\
P2 & 41 & 130 & No\\
P3 & 31 & 145 & No\\
P4 & 45 & 111 & Yes\\
P5 & 56 & 123 & No\\
P6 & 65 & 321 & Yes
\end{bmatrix}
")

Note that the matrix
![\\boldsymbol{X} \\in \\mathbb{R}^{6 \\times 4}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B6%20%5Ctimes%204%7D "\boldsymbol{X} \in \mathbb{R}^{6 \times 4}").Further,
we write row 2, row 4, row 6 of
![\\boldsymbol{X}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Cboldsymbol%7BX%7D "\boldsymbol{X}")
which will be denoted by vectors
![x_2, x_4, x_6](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_2%2C%20x_4%2C%20x_6 "x_2, x_4, x_6")
respectively:

![x_2 = \\begin{pmatrix}
P2 \\\\ 41 \\\\ 130 \\\\ No
\\end{pmatrix}; x_4 = \\begin{pmatrix} P4 \\\\ 45 \\\\ 111 \\\\ Yes \\end{pmatrix}; x_6 = \\begin{pmatrix} P6 \\\\ 65 \\\\ 321 \\\\ Yes \\end{pmatrix}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_2%20%3D%20%5Cbegin%7Bpmatrix%7D%0AP2%20%5C%5C%2041%20%5C%5C%20130%20%5C%5C%20No%0A%5Cend%7Bpmatrix%7D%3B%20x_4%20%3D%20%5Cbegin%7Bpmatrix%7D%20P4%20%5C%5C%2045%20%5C%5C%20111%20%5C%5C%20Yes%20%5Cend%7Bpmatrix%7D%3B%20x_6%20%3D%20%5Cbegin%7Bpmatrix%7D%20P6%20%5C%5C%2065%20%5C%5C%20321%20%5C%5C%20Yes%20%5Cend%7Bpmatrix%7D "x_2 = \begin{pmatrix}
P2 \\ 41 \\ 130 \\ No
\end{pmatrix}; x_4 = \begin{pmatrix} P4 \\ 45 \\ 111 \\ Yes \end{pmatrix}; x_6 = \begin{pmatrix} P6 \\ 65 \\ 321 \\ Yes \end{pmatrix}")

Again,
![x_i \\in \\mathbb{R}^4](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;x_i%20%5Cin%20%5Cmathbb%7BR%7D%5E4 "x_i \in \mathbb{R}^4").
Similarly, we can write the column(s) 1, 3, 4 which are vectors of
length
![n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n "n")
denoted by
![\\textbf{x}\_1, \\textbf{x}\_3, \\textbf{x}\_4](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7Bx%7D_1%2C%20%5Ctextbf%7Bx%7D_3%2C%20%5Ctextbf%7Bx%7D_4 "\textbf{x}_1, \textbf{x}_3, \textbf{x}_4")
respectively:

![\\textbf{x}\_1 = \\begin{pmatrix}
P1 \\\\ P2 \\\\ P3 \\\\ P4 \\\\ P5 \\\\ P6
\\end{pmatrix}; \\textbf{x}\_3 = \\begin{pmatrix} 110 \\\\ 130 \\\\ 145\\\\111 \\\\ 123\\\\ 321 \\end{pmatrix}; \\textbf{x}\_4 = \\begin{pmatrix} Yes \\\\ No \\\\ No \\\\ Yes \\\\No \\\\Yes \\end{pmatrix}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7Bx%7D_1%20%3D%20%5Cbegin%7Bpmatrix%7D%0AP1%20%5C%5C%20P2%20%5C%5C%20P3%20%5C%5C%20P4%20%5C%5C%20P5%20%5C%5C%20P6%0A%5Cend%7Bpmatrix%7D%3B%20%5Ctextbf%7Bx%7D_3%20%3D%20%5Cbegin%7Bpmatrix%7D%20110%20%5C%5C%20130%20%5C%5C%20145%5C%5C111%20%5C%5C%20123%5C%5C%20321%20%5Cend%7Bpmatrix%7D%3B%20%5Ctextbf%7Bx%7D_4%20%3D%20%5Cbegin%7Bpmatrix%7D%20Yes%20%5C%5C%20No%20%5C%5C%20No%20%5C%5C%20Yes%20%5C%5CNo%20%5C%5CYes%20%5Cend%7Bpmatrix%7D "\textbf{x}_1 = \begin{pmatrix}
P1 \\ P2 \\ P3 \\ P4 \\ P5 \\ P6
\end{pmatrix}; \textbf{x}_3 = \begin{pmatrix} 110 \\ 130 \\ 145\\111 \\ 123\\ 321 \end{pmatrix}; \textbf{x}_4 = \begin{pmatrix} Yes \\ No \\ No \\ Yes \\No \\Yes \end{pmatrix}")

![\\textbf{x} \\in \\mathbb{R}^6](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7Bx%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E6 "\textbf{x} \in \mathbb{R}^6").
Finally, if we ask Machine Learning to predict whether or not a patient
has diabetes based on a set of factors or columns, like *Age*, *Glucose
Lvl*, then we can denote
![\\textbf{y}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7By%7D "\textbf{y}")
as the column or variable *Diabetic* mostly because that is the result
we’re after, whether or not diabetes is present.

![\\textbf{y} = \\begin{pmatrix}
Yes\\\\ No \\\\ No \\\\ Yes \\\\No \\\\Yes
\\end{pmatrix}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctextbf%7By%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%0AYes%5C%5C%20No%20%5C%5C%20No%20%5C%5C%20Yes%20%5C%5CNo%20%5C%5CYes%0A%5Cend%7Bpmatrix%7D "\textbf{y} = \begin{pmatrix}
Yes\\ No \\ No \\ Yes \\No \\Yes
\end{pmatrix}")

The next notebook will be a more in-depth look into Statistical
Learning, where topics such as Machine Learning (ML) jargon, ML system
classification, and more will be covered.
