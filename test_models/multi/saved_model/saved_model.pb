��#
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02unknown8�� 
p
output1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput1/bias
i
 output1/bias/Read/ReadVariableOpReadVariableOpoutput1/bias*
_output_shapes
:*
dtype0
y
output1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameoutput1/kernel
r
"output1/kernel/Read/ReadVariableOpReadVariableOpoutput1/kernel*
_output_shapes
:	�*
dtype0
�
#batch_normalization_5/renorm_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_5/renorm_stddev
�
7batch_normalization_5/renorm_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_5/renorm_stddev*
_output_shapes	
:�*
dtype0
�
!batch_normalization_5/renorm_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_5/renorm_mean
�
5batch_normalization_5/renorm_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/renorm_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization_5/moving_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_5/moving_stddev
�
7batch_normalization_5/moving_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_5/moving_stddev*
_output_shapes	
:�*
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:�*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:�*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
��*
dtype0
�
#batch_normalization_4/renorm_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_4/renorm_stddev
�
7batch_normalization_4/renorm_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_4/renorm_stddev*
_output_shapes	
:�*
dtype0
�
!batch_normalization_4/renorm_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_4/renorm_mean
�
5batch_normalization_4/renorm_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/renorm_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization_4/moving_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_4/moving_stddev
�
7batch_normalization_4/moving_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_4/moving_stddev*
_output_shapes	
:�*
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
��*
dtype0
�
#batch_normalization_3/renorm_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_3/renorm_stddev
�
7batch_normalization_3/renorm_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_3/renorm_stddev*
_output_shapes	
:�*
dtype0
�
!batch_normalization_3/renorm_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_3/renorm_mean
�
5batch_normalization_3/renorm_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/renorm_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization_3/moving_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_3/moving_stddev
�
7batch_normalization_3/moving_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_3/moving_stddev*
_output_shapes	
:�*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
�
#batch_normalization_2/renorm_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_2/renorm_stddev
�
7batch_normalization_2/renorm_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_2/renorm_stddev*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/renorm_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/renorm_mean
�
5batch_normalization_2/renorm_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/renorm_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization_2/moving_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_2/moving_stddev
�
7batch_normalization_2/moving_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_2/moving_stddev*
_output_shapes	
:�*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
�
#batch_normalization_1/renorm_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_1/renorm_stddev
�
7batch_normalization_1/renorm_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_1/renorm_stddev*
_output_shapes	
:�*
dtype0
�
!batch_normalization_1/renorm_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_1/renorm_mean
�
5batch_normalization_1/renorm_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/renorm_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization_1/moving_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_1/moving_stddev
�
7batch_normalization_1/moving_stddev/Read/ReadVariableOpReadVariableOp#batch_normalization_1/moving_stddev*
_output_shapes	
:�*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
�
!batch_normalization/renorm_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization/renorm_stddev
�
5batch_normalization/renorm_stddev/Read/ReadVariableOpReadVariableOp!batch_normalization/renorm_stddev*
_output_shapes
:*
dtype0
�
batch_normalization/renorm_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/renorm_mean
�
3batch_normalization/renorm_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/renorm_mean*
_output_shapes
:*
dtype0
�
!batch_normalization/moving_stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization/moving_stddev
�
5batch_normalization/moving_stddev/Read/ReadVariableOpReadVariableOp!batch_normalization/moving_stddev*
_output_shapes
:*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
y
serving_default_input1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
y
serving_default_input2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�

StatefulPartitionedCallStatefulPartitionedCallserving_default_input1serving_default_input2#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_1/kerneldense_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_2/kerneldense_2/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betadense_3/kerneldense_3/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betadense_4/kerneldense_4/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betaoutput1/kerneloutput1/bias*1
Tin*
(2&*
Tout
2
*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_2678

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer-15
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&renorm_clipping
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,moving_stddev
-renorm_mean
.renorm_stddev*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=renorm_clipping
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Cmoving_stddev
Drenorm_mean
Erenorm_stddev*
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Trenorm_clipping
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Zmoving_stddev
[renorm_mean
\renorm_stddev*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
krenorm_clipping
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qmoving_stddev
rrenorm_mean
srenorm_stddev*
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias*
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�renorm_clipping
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�moving_stddev
�renorm_mean
�renorm_stddev*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�renorm_clipping
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�moving_stddev
�renorm_mean
�renorm_stddev*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
(0
)1
*2
+3
,4
-5
.6
57
68
?9
@10
A11
B12
C13
D14
E15
L16
M17
V18
W19
X20
Y21
Z22
[23
\24
c25
d26
m27
n28
o29
p30
q31
r32
s33
z34
{35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53*
�
(0
)1
52
63
?4
@5
L6
M7
V8
W9
c10
d11
m12
n13
z14
{15
�16
�17
�18
�19
�20
�21
�22
�23*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
5
(0
)1
*2
+3
,4
-5
.6*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE!batch_normalization/moving_stddev=layer_with_weights-0/moving_stddev/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/renorm_mean;layer_with_weights-0/renorm_mean/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE!batch_normalization/renorm_stddev=layer_with_weights-0/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
5
?0
@1
A2
B3
C4
D5
E6*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_1/moving_stddev=layer_with_weights-2/moving_stddev/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/renorm_mean;layer_with_weights-2/renorm_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_1/renorm_stddev=layer_with_weights-2/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
5
V0
W1
X2
Y3
Z4
[5
\6*

V0
W1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_2/moving_stddev=layer_with_weights-4/moving_stddev/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/renorm_mean;layer_with_weights-4/renorm_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_2/renorm_stddev=layer_with_weights-4/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

c0
d1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
5
m0
n1
o2
p3
q4
r5
s6*

m0
n1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_3/moving_stddev=layer_with_weights-6/moving_stddev/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/renorm_mean;layer_with_weights-6/renorm_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_3/renorm_stddev=layer_with_weights-6/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUE*

z0
{1*

z0
{1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
<
�0
�1
�2
�3
�4
�5
�6*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_4/moving_stddev=layer_with_weights-8/moving_stddev/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/renorm_mean;layer_with_weights-8/renorm_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE#batch_normalization_4/renorm_stddev=layer_with_weights-8/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
<
�0
�1
�2
�3
�4
�5
�6*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE#batch_normalization_5/moving_stddev>layer_with_weights-10/moving_stddev/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_5/renorm_mean<layer_with_weights-10/renorm_mean/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE#batch_normalization_5/renorm_stddev>layer_with_weights-10/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEoutput1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEoutput1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
*0
+1
,2
-3
.4
A5
B6
C7
D8
E9
X10
Y11
Z12
[13
\14
o15
p16
q17
r18
s19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
'
*0
+1
,2
-3
.4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
'
A0
B1
C2
D3
E4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
'
X0
Y1
Z2
[3
\4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
'
o0
p1
q2
r3
s4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
,
�0
�1
�2
�3
�4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
,
�0
�1
�2
�3
�4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization/moving_stddevbatch_normalization/renorm_mean!batch_normalization/renorm_stddevdense/kernel
dense/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance#batch_normalization_1/moving_stddev!batch_normalization_1/renorm_mean#batch_normalization_1/renorm_stddevdense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#batch_normalization_2/moving_stddev!batch_normalization_2/renorm_mean#batch_normalization_2/renorm_stddevdense_2/kerneldense_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#batch_normalization_3/moving_stddev!batch_normalization_3/renorm_mean#batch_normalization_3/renorm_stddevdense_3/kerneldense_3/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#batch_normalization_4/moving_stddev!batch_normalization_4/renorm_mean#batch_normalization_4/renorm_stddevdense_4/kerneldense_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance#batch_normalization_5/moving_stddev!batch_normalization_5/renorm_mean#batch_normalization_5/renorm_stddevoutput1/kerneloutput1/biasConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_4803
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization/moving_stddevbatch_normalization/renorm_mean!batch_normalization/renorm_stddevdense/kernel
dense/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance#batch_normalization_1/moving_stddev!batch_normalization_1/renorm_mean#batch_normalization_1/renorm_stddevdense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#batch_normalization_2/moving_stddev!batch_normalization_2/renorm_mean#batch_normalization_2/renorm_stddevdense_2/kerneldense_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#batch_normalization_3/moving_stddev!batch_normalization_3/renorm_mean#batch_normalization_3/renorm_stddevdense_3/kerneldense_3/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#batch_normalization_4/moving_stddev!batch_normalization_4/renorm_mean#batch_normalization_4/renorm_stddevdense_4/kerneldense_4/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance#batch_normalization_5/moving_stddev!batch_normalization_5/renorm_mean#batch_normalization_5/renorm_stddevoutput1/kerneloutput1/bias*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_4975��
�

�
A__inference_dense_4_layer_call_and_return_conditional_losses_1727

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4191

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1477

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4417

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_2_layer_call_fn_3873

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1174p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
?__inference_dense_layer_call_and_return_conditional_losses_3692

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
]
A__inference_output2_layer_call_and_return_conditional_losses_1773

inputs

identity
I
ShapeShapeinputs*
T0
*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0
*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0
*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_dense_1_layer_call_and_return_conditional_losses_3841

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_4_layer_call_fn_4152

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1346p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�]
�
?__inference_model_layer_call_and_return_conditional_losses_2596

input1

input2&
batch_normalization_2471:&
batch_normalization_2473:&
batch_normalization_2475:&
batch_normalization_2477:&
batch_normalization_2479:&
batch_normalization_2481:&
batch_normalization_2483:

dense_2486:	�

dense_2488:	�)
batch_normalization_1_2491:	�)
batch_normalization_1_2493:	�)
batch_normalization_1_2495:	�)
batch_normalization_1_2497:	�)
batch_normalization_1_2499:	�)
batch_normalization_1_2501:	�)
batch_normalization_1_2503:	� 
dense_1_2506:
��
dense_1_2508:	�)
batch_normalization_2_2511:	�)
batch_normalization_2_2513:	�)
batch_normalization_2_2515:	�)
batch_normalization_2_2517:	�)
batch_normalization_2_2519:	�)
batch_normalization_2_2521:	�)
batch_normalization_2_2523:	� 
dense_2_2526:
��
dense_2_2528:	�)
batch_normalization_3_2531:	�)
batch_normalization_3_2533:	�)
batch_normalization_3_2535:	�)
batch_normalization_3_2537:	�)
batch_normalization_3_2539:	�)
batch_normalization_3_2541:	�)
batch_normalization_3_2543:	� 
dense_3_2546:
��
dense_3_2548:	�)
batch_normalization_4_2551:	�)
batch_normalization_4_2553:	�)
batch_normalization_4_2555:	�)
batch_normalization_4_2557:	�)
batch_normalization_4_2559:	�)
batch_normalization_4_2561:	�)
batch_normalization_4_2563:	� 
dense_4_2566:
��
dense_4_2568:	�)
batch_normalization_5_2571:	�)
batch_normalization_5_2573:	�)
batch_normalization_5_2575:	�)
batch_normalization_5_2577:	�)
batch_normalization_5_2579:	�)
batch_normalization_5_2581:	�)
batch_normalization_5_2583:	�
output1_2586:	�
output1_2588:
identity

identity_1
��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�output1/StatefulPartitionedCalla
concatenate/CastCastinput2*

DstT0*

SrcT0*'
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCallinput1concatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1601�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_2471batch_normalization_2473batch_normalization_2475batch_normalization_2477batch_normalization_2479batch_normalization_2481batch_normalization_2483*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_912�
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
dense_2486
dense_2488*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1623�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_2491batch_normalization_1_2493batch_normalization_1_2495batch_normalization_1_2497batch_normalization_1_2499batch_normalization_1_2501batch_normalization_1_2503*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1043�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_2506dense_1_2508*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1649�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_2511batch_normalization_2_2513batch_normalization_2_2515batch_normalization_2_2517batch_normalization_2_2519batch_normalization_2_2521batch_normalization_2_2523*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1174�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_2526dense_2_2528*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1675�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_3_2531batch_normalization_3_2533batch_normalization_3_2535batch_normalization_3_2537batch_normalization_3_2539batch_normalization_3_2541batch_normalization_3_2543*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1305�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_3_2546dense_3_2548*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1701�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_4_2551batch_normalization_4_2553batch_normalization_4_2555batch_normalization_4_2557batch_normalization_4_2559batch_normalization_4_2561batch_normalization_4_2563*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1436�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_4_2566dense_4_2568*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1727�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_5_2571batch_normalization_5_2573batch_normalization_5_2575batch_normalization_5_2577batch_normalization_5_2579batch_normalization_5_2581batch_normalization_5_2583*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1567�
output1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0output1_2586output1_2588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output1_layer_call_and_return_conditional_losses_1753^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater/GreaterGreater(output1/StatefulPartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*'
_output_shapes
:����������
output2/PartitionedCallPartitionedCalltf.math.greater/Greater:z:0*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output2_layer_call_and_return_conditional_losses_1773w
IdentityIdentity(output1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity output2/PartitionedCall:output:0^NoOp*
T0
*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^output1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinput1:OK
'
_output_shapes
:���������
 
_user_specified_nameinput2
�
�
&__inference_dense_2_layer_call_fn_3979

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1675p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1346

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
V
*__inference_concatenate_layer_call_fn_3536
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1601`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1084

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�P
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3672

inputs-
maximum_readvariableop_resource:)
sub_readvariableop_resource:)
mul_readvariableop_resource:+
add_1_readvariableop_resource:7
)assignmovingavg_2_readvariableop_resource:7
)assignmovingavg_3_readvariableop_resource:%
assignnewvalue_resource:

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:]
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes
::
SqrtSqrtadd:z:0*
T0*
_output_shapes
:M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: r
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
:*
dtype0c
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes
:v
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:j
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0e
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes
:w
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:i
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes
:[

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes
:N

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes
:Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<v
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0|
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0P

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes
:\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<|
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0t

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes
:}

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes
:J
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes
:L
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes
:j
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:*
dtype0^
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:*
dtype0b
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0\
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes
:\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Z
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes
:>
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes
:\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes
:*
dtype0w
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes
:~
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes
:*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes
:*
dtype0g
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:N
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes
:<
ReluRelu	sub_1:z:0*
T0*
_output_shapes
:�
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:W
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Y
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������d

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_5_layer_call_fn_4301

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1477p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4268

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_2678

input1

input2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:	�

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:
identity

identity_1
��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput1input2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2
*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__wrapped_model_798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0
*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinput1:OK
'
_output_shapes
:���������
 
_user_specified_nameinput2
��
� 
?__inference_model_layer_call_and_return_conditional_losses_3031
inputs_0
inputs_1C
5batch_normalization_batchnorm_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:E
7batch_normalization_batchnorm_readvariableop_1_resource:E
7batch_normalization_batchnorm_readvariableop_2_resource:7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�F
7batch_normalization_2_batchnorm_readvariableop_resource:	�J
;batch_normalization_2_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_2_batchnorm_readvariableop_2_resource:	�:
&dense_2_matmul_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�F
7batch_normalization_3_batchnorm_readvariableop_resource:	�J
;batch_normalization_3_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_3_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_3_batchnorm_readvariableop_2_resource:	�:
&dense_3_matmul_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�F
7batch_normalization_4_batchnorm_readvariableop_resource:	�J
;batch_normalization_4_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_4_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_4_batchnorm_readvariableop_2_resource:	�:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�F
7batch_normalization_5_batchnorm_readvariableop_resource:	�J
;batch_normalization_5_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_5_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_5_batchnorm_readvariableop_2_resource:	�9
&output1_matmul_readvariableop_resource:	�5
'output1_biasadd_readvariableop_resource:
identity

identity_1
��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�.batch_normalization_3/batchnorm/ReadVariableOp�0batch_normalization_3/batchnorm/ReadVariableOp_1�0batch_normalization_3/batchnorm/ReadVariableOp_2�2batch_normalization_3/batchnorm/mul/ReadVariableOp�.batch_normalization_4/batchnorm/ReadVariableOp�0batch_normalization_4/batchnorm/ReadVariableOp_1�0batch_normalization_4/batchnorm/ReadVariableOp_2�2batch_normalization_4/batchnorm/mul/ReadVariableOp�.batch_normalization_5/batchnorm/ReadVariableOp�0batch_normalization_5/batchnorm/ReadVariableOp_1�0batch_normalization_5/batchnorm/ReadVariableOp_2�2batch_normalization_5/batchnorm/mul/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�output1/BiasAdd/ReadVariableOp�output1/MatMul/ReadVariableOpc
concatenate/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2inputs_0concatenate/Cast:y:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Mulconcatenate/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/mul_1Muldense/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_2/batchnorm/mul_1Muldense_1/Tanh:y:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/mul_1Muldense_2/Tanh:y:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_3/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/mul_1Muldense_3/Tanh:y:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_4/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/mul_1Muldense_4/Tanh:y:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
output1/MatMul/ReadVariableOpReadVariableOp&output1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output1/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%output1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output1/BiasAdd/ReadVariableOpReadVariableOp'output1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output1/BiasAddBiasAddoutput1/MatMul:product:0&output1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
output1/SoftmaxSoftmaxoutput1/BiasAdd:output:0*
T0*'
_output_shapes
:���������^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater/GreaterGreateroutput1/Softmax:softmax:0"tf.math.greater/Greater/y:output:0*
T0*'
_output_shapes
:���������f
output2/ShapeShapetf.math.greater/Greater:z:0*
T0
*
_output_shapes
::��e
output2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
output2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
output2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
output2/strided_sliceStridedSliceoutput2/Shape:output:0$output2/strided_slice/stack:output:0&output2/strided_slice/stack_1:output:0&output2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
output2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
output2/Reshape/shapePackoutput2/strided_slice:output:0 output2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
output2/ReshapeReshapetf.math.greater/Greater:z:0output2/Reshape/shape:output:0*
T0
*'
_output_shapes
:���������h
IdentityIdentityoutput1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������i

Identity_1Identityoutput2/Reshape:output:0^NoOp*
T0
*'
_output_shapes
:����������
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
output1/BiasAdd/ReadVariableOpoutput1/BiasAdd/ReadVariableOp2>
output1/MatMul/ReadVariableOpoutput1/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1
�
�
&__inference_output1_layer_call_fn_4426

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output1_layer_call_and_return_conditional_losses_1753o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_1_layer_call_fn_3724

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1043p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�4
__inference__traced_save_4803
file_prefix>
0read_disablecopyonread_batch_normalization_gamma:?
1read_1_disablecopyonread_batch_normalization_beta:F
8read_2_disablecopyonread_batch_normalization_moving_mean:J
<read_3_disablecopyonread_batch_normalization_moving_variance:H
:read_4_disablecopyonread_batch_normalization_moving_stddev:F
8read_5_disablecopyonread_batch_normalization_renorm_mean:H
:read_6_disablecopyonread_batch_normalization_renorm_stddev:8
%read_7_disablecopyonread_dense_kernel:	�2
#read_8_disablecopyonread_dense_bias:	�C
4read_9_disablecopyonread_batch_normalization_1_gamma:	�C
4read_10_disablecopyonread_batch_normalization_1_beta:	�J
;read_11_disablecopyonread_batch_normalization_1_moving_mean:	�N
?read_12_disablecopyonread_batch_normalization_1_moving_variance:	�L
=read_13_disablecopyonread_batch_normalization_1_moving_stddev:	�J
;read_14_disablecopyonread_batch_normalization_1_renorm_mean:	�L
=read_15_disablecopyonread_batch_normalization_1_renorm_stddev:	�<
(read_16_disablecopyonread_dense_1_kernel:
��5
&read_17_disablecopyonread_dense_1_bias:	�D
5read_18_disablecopyonread_batch_normalization_2_gamma:	�C
4read_19_disablecopyonread_batch_normalization_2_beta:	�J
;read_20_disablecopyonread_batch_normalization_2_moving_mean:	�N
?read_21_disablecopyonread_batch_normalization_2_moving_variance:	�L
=read_22_disablecopyonread_batch_normalization_2_moving_stddev:	�J
;read_23_disablecopyonread_batch_normalization_2_renorm_mean:	�L
=read_24_disablecopyonread_batch_normalization_2_renorm_stddev:	�<
(read_25_disablecopyonread_dense_2_kernel:
��5
&read_26_disablecopyonread_dense_2_bias:	�D
5read_27_disablecopyonread_batch_normalization_3_gamma:	�C
4read_28_disablecopyonread_batch_normalization_3_beta:	�J
;read_29_disablecopyonread_batch_normalization_3_moving_mean:	�N
?read_30_disablecopyonread_batch_normalization_3_moving_variance:	�L
=read_31_disablecopyonread_batch_normalization_3_moving_stddev:	�J
;read_32_disablecopyonread_batch_normalization_3_renorm_mean:	�L
=read_33_disablecopyonread_batch_normalization_3_renorm_stddev:	�<
(read_34_disablecopyonread_dense_3_kernel:
��5
&read_35_disablecopyonread_dense_3_bias:	�D
5read_36_disablecopyonread_batch_normalization_4_gamma:	�C
4read_37_disablecopyonread_batch_normalization_4_beta:	�J
;read_38_disablecopyonread_batch_normalization_4_moving_mean:	�N
?read_39_disablecopyonread_batch_normalization_4_moving_variance:	�L
=read_40_disablecopyonread_batch_normalization_4_moving_stddev:	�J
;read_41_disablecopyonread_batch_normalization_4_renorm_mean:	�L
=read_42_disablecopyonread_batch_normalization_4_renorm_stddev:	�<
(read_43_disablecopyonread_dense_4_kernel:
��5
&read_44_disablecopyonread_dense_4_bias:	�D
5read_45_disablecopyonread_batch_normalization_5_gamma:	�C
4read_46_disablecopyonread_batch_normalization_5_beta:	�J
;read_47_disablecopyonread_batch_normalization_5_moving_mean:	�N
?read_48_disablecopyonread_batch_normalization_5_moving_variance:	�L
=read_49_disablecopyonread_batch_normalization_5_moving_stddev:	�J
;read_50_disablecopyonread_batch_normalization_5_renorm_mean:	�L
=read_51_disablecopyonread_batch_normalization_5_renorm_stddev:	�;
(read_52_disablecopyonread_output1_kernel:	�4
&read_53_disablecopyonread_output1_bias:
savev2_const
identity_109��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead0read_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp0read_disablecopyonread_batch_normalization_gamma^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_1/DisableCopyOnReadDisableCopyOnRead1read_1_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp1read_1_disablecopyonread_batch_normalization_beta^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead8read_2_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp8read_2_disablecopyonread_batch_normalization_moving_mean^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead<read_3_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp<read_3_disablecopyonread_batch_normalization_moving_variance^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead:read_4_disablecopyonread_batch_normalization_moving_stddev"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp:read_4_disablecopyonread_batch_normalization_moving_stddev^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead8read_5_disablecopyonread_batch_normalization_renorm_mean"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp8read_5_disablecopyonread_batch_normalization_renorm_mean^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead:read_6_disablecopyonread_batch_normalization_renorm_stddev"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp:read_6_disablecopyonread_batch_normalization_renorm_stddev^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	�w
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_dense_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_batch_normalization_1_gamma^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnRead4read_10_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp4read_10_disablecopyonread_batch_normalization_1_beta^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnRead;read_11_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp;read_11_disablecopyonread_batch_normalization_1_moving_mean^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnRead?read_12_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp?read_12_disablecopyonread_batch_normalization_1_moving_variance^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_13/DisableCopyOnReadDisableCopyOnRead=read_13_disablecopyonread_batch_normalization_1_moving_stddev"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp=read_13_disablecopyonread_batch_normalization_1_moving_stddev^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnRead;read_14_disablecopyonread_batch_normalization_1_renorm_mean"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp;read_14_disablecopyonread_batch_normalization_1_renorm_mean^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnRead=read_15_disablecopyonread_batch_normalization_1_renorm_stddev"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp=read_15_disablecopyonread_batch_normalization_1_renorm_stddev^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_dense_1_kernel^Read_16/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_17/DisableCopyOnReadDisableCopyOnRead&read_17_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp&read_17_disablecopyonread_dense_1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead5read_18_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp5read_18_disablecopyonread_batch_normalization_2_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnRead4read_19_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp4read_19_disablecopyonread_batch_normalization_2_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead;read_20_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp;read_20_disablecopyonread_batch_normalization_2_moving_mean^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead?read_21_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp?read_21_disablecopyonread_batch_normalization_2_moving_variance^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead=read_22_disablecopyonread_batch_normalization_2_moving_stddev"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp=read_22_disablecopyonread_batch_normalization_2_moving_stddev^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_23/DisableCopyOnReadDisableCopyOnRead;read_23_disablecopyonread_batch_normalization_2_renorm_mean"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp;read_23_disablecopyonread_batch_normalization_2_renorm_mean^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead=read_24_disablecopyonread_batch_normalization_2_renorm_stddev"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp=read_24_disablecopyonread_batch_normalization_2_renorm_stddev^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_25/DisableCopyOnReadDisableCopyOnRead(read_25_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp(read_25_disablecopyonread_dense_2_kernel^Read_25/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_26/DisableCopyOnReadDisableCopyOnRead&read_26_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp&read_26_disablecopyonread_dense_2_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_27/DisableCopyOnReadDisableCopyOnRead5read_27_disablecopyonread_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp5read_27_disablecopyonread_batch_normalization_3_gamma^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnRead4read_28_disablecopyonread_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp4read_28_disablecopyonread_batch_normalization_3_beta^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead;read_29_disablecopyonread_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp;read_29_disablecopyonread_batch_normalization_3_moving_mean^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead?read_30_disablecopyonread_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp?read_30_disablecopyonread_batch_normalization_3_moving_variance^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead=read_31_disablecopyonread_batch_normalization_3_moving_stddev"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp=read_31_disablecopyonread_batch_normalization_3_moving_stddev^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnRead;read_32_disablecopyonread_batch_normalization_3_renorm_mean"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp;read_32_disablecopyonread_batch_normalization_3_renorm_mean^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnRead=read_33_disablecopyonread_batch_normalization_3_renorm_stddev"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp=read_33_disablecopyonread_batch_normalization_3_renorm_stddev^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_34/DisableCopyOnReadDisableCopyOnRead(read_34_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp(read_34_disablecopyonread_dense_3_kernel^Read_34/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_35/DisableCopyOnReadDisableCopyOnRead&read_35_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp&read_35_disablecopyonread_dense_3_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_36/DisableCopyOnReadDisableCopyOnRead5read_36_disablecopyonread_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp5read_36_disablecopyonread_batch_normalization_4_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_37/DisableCopyOnReadDisableCopyOnRead4read_37_disablecopyonread_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp4read_37_disablecopyonread_batch_normalization_4_beta^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_38/DisableCopyOnReadDisableCopyOnRead;read_38_disablecopyonread_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp;read_38_disablecopyonread_batch_normalization_4_moving_mean^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_39/DisableCopyOnReadDisableCopyOnRead?read_39_disablecopyonread_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp?read_39_disablecopyonread_batch_normalization_4_moving_variance^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_40/DisableCopyOnReadDisableCopyOnRead=read_40_disablecopyonread_batch_normalization_4_moving_stddev"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp=read_40_disablecopyonread_batch_normalization_4_moving_stddev^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_41/DisableCopyOnReadDisableCopyOnRead;read_41_disablecopyonread_batch_normalization_4_renorm_mean"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp;read_41_disablecopyonread_batch_normalization_4_renorm_mean^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead=read_42_disablecopyonread_batch_normalization_4_renorm_stddev"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp=read_42_disablecopyonread_batch_normalization_4_renorm_stddev^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_43/DisableCopyOnReadDisableCopyOnRead(read_43_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp(read_43_disablecopyonread_dense_4_kernel^Read_43/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_44/DisableCopyOnReadDisableCopyOnRead&read_44_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp&read_44_disablecopyonread_dense_4_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_45/DisableCopyOnReadDisableCopyOnRead5read_45_disablecopyonread_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp5read_45_disablecopyonread_batch_normalization_5_gamma^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_46/DisableCopyOnReadDisableCopyOnRead4read_46_disablecopyonread_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp4read_46_disablecopyonread_batch_normalization_5_beta^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_47/DisableCopyOnReadDisableCopyOnRead;read_47_disablecopyonread_batch_normalization_5_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp;read_47_disablecopyonread_batch_normalization_5_moving_mean^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_48/DisableCopyOnReadDisableCopyOnRead?read_48_disablecopyonread_batch_normalization_5_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp?read_48_disablecopyonread_batch_normalization_5_moving_variance^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnRead=read_49_disablecopyonread_batch_normalization_5_moving_stddev"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp=read_49_disablecopyonread_batch_normalization_5_moving_stddev^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnRead;read_50_disablecopyonread_batch_normalization_5_renorm_mean"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp;read_50_disablecopyonread_batch_normalization_5_renorm_mean^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnRead=read_51_disablecopyonread_batch_normalization_5_renorm_stddev"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp=read_51_disablecopyonread_batch_normalization_5_renorm_stddev^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_52/DisableCopyOnReadDisableCopyOnRead(read_52_disablecopyonread_output1_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp(read_52_disablecopyonread_output1_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_53/DisableCopyOnReadDisableCopyOnRead&read_53_disablecopyonread_output1_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp&read_53_disablecopyonread_output1_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-0/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-0/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-4/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-4/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-8/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-8/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_108Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_109IdentityIdentity_108:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_109Identity_109:output:0*�
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:7

_output_shapes
: 
�

�
?__inference_dense_layer_call_and_return_conditional_losses_1623

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_953

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�%
 __inference__traced_restore_4975
file_prefix8
*assignvariableop_batch_normalization_gamma:9
+assignvariableop_1_batch_normalization_beta:@
2assignvariableop_2_batch_normalization_moving_mean:D
6assignvariableop_3_batch_normalization_moving_variance:B
4assignvariableop_4_batch_normalization_moving_stddev:@
2assignvariableop_5_batch_normalization_renorm_mean:B
4assignvariableop_6_batch_normalization_renorm_stddev:2
assignvariableop_7_dense_kernel:	�,
assignvariableop_8_dense_bias:	�=
.assignvariableop_9_batch_normalization_1_gamma:	�=
.assignvariableop_10_batch_normalization_1_beta:	�D
5assignvariableop_11_batch_normalization_1_moving_mean:	�H
9assignvariableop_12_batch_normalization_1_moving_variance:	�F
7assignvariableop_13_batch_normalization_1_moving_stddev:	�D
5assignvariableop_14_batch_normalization_1_renorm_mean:	�F
7assignvariableop_15_batch_normalization_1_renorm_stddev:	�6
"assignvariableop_16_dense_1_kernel:
��/
 assignvariableop_17_dense_1_bias:	�>
/assignvariableop_18_batch_normalization_2_gamma:	�=
.assignvariableop_19_batch_normalization_2_beta:	�D
5assignvariableop_20_batch_normalization_2_moving_mean:	�H
9assignvariableop_21_batch_normalization_2_moving_variance:	�F
7assignvariableop_22_batch_normalization_2_moving_stddev:	�D
5assignvariableop_23_batch_normalization_2_renorm_mean:	�F
7assignvariableop_24_batch_normalization_2_renorm_stddev:	�6
"assignvariableop_25_dense_2_kernel:
��/
 assignvariableop_26_dense_2_bias:	�>
/assignvariableop_27_batch_normalization_3_gamma:	�=
.assignvariableop_28_batch_normalization_3_beta:	�D
5assignvariableop_29_batch_normalization_3_moving_mean:	�H
9assignvariableop_30_batch_normalization_3_moving_variance:	�F
7assignvariableop_31_batch_normalization_3_moving_stddev:	�D
5assignvariableop_32_batch_normalization_3_renorm_mean:	�F
7assignvariableop_33_batch_normalization_3_renorm_stddev:	�6
"assignvariableop_34_dense_3_kernel:
��/
 assignvariableop_35_dense_3_bias:	�>
/assignvariableop_36_batch_normalization_4_gamma:	�=
.assignvariableop_37_batch_normalization_4_beta:	�D
5assignvariableop_38_batch_normalization_4_moving_mean:	�H
9assignvariableop_39_batch_normalization_4_moving_variance:	�F
7assignvariableop_40_batch_normalization_4_moving_stddev:	�D
5assignvariableop_41_batch_normalization_4_renorm_mean:	�F
7assignvariableop_42_batch_normalization_4_renorm_stddev:	�6
"assignvariableop_43_dense_4_kernel:
��/
 assignvariableop_44_dense_4_bias:	�>
/assignvariableop_45_batch_normalization_5_gamma:	�=
.assignvariableop_46_batch_normalization_5_beta:	�D
5assignvariableop_47_batch_normalization_5_moving_mean:	�H
9assignvariableop_48_batch_normalization_5_moving_variance:	�F
7assignvariableop_49_batch_normalization_5_moving_stddev:	�D
5assignvariableop_50_batch_normalization_5_renorm_mean:	�F
7assignvariableop_51_batch_normalization_5_renorm_stddev:	�5
"assignvariableop_52_output1_kernel:	�.
 assignvariableop_53_output1_bias:
identity_55��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-0/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-0/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-2/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-4/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-4/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-6/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-8/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB=layer_with_weights-8/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/moving_stddev/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/renorm_mean/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-10/renorm_stddev/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_moving_stddevIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp2assignvariableop_5_batch_normalization_renorm_meanIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_renorm_stddevIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_1_gammaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp.assignvariableop_10_batch_normalization_1_betaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp5assignvariableop_11_batch_normalization_1_moving_meanIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp9assignvariableop_12_batch_normalization_1_moving_varianceIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_1_moving_stddevIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_1_renorm_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp7assignvariableop_15_batch_normalization_1_renorm_stddevIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_2_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_2_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_2_moving_meanIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_2_moving_varianceIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_2_moving_stddevIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_2_renorm_meanIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp7assignvariableop_24_batch_normalization_2_renorm_stddevIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_dense_2_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp assignvariableop_26_dense_2_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_3_gammaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_batch_normalization_3_betaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp5assignvariableop_29_batch_normalization_3_moving_meanIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp9assignvariableop_30_batch_normalization_3_moving_varianceIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_3_moving_stddevIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_3_renorm_meanIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp7assignvariableop_33_batch_normalization_3_renorm_stddevIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_3_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_3_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_4_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_4_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_4_moving_meanIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_4_moving_varianceIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp7assignvariableop_40_batch_normalization_4_moving_stddevIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp5assignvariableop_41_batch_normalization_4_renorm_meanIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp7assignvariableop_42_batch_normalization_4_renorm_stddevIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp"assignvariableop_43_dense_4_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp assignvariableop_44_dense_4_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_5_gammaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp.assignvariableop_46_batch_normalization_5_betaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp5assignvariableop_47_batch_normalization_5_moving_meanIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp9assignvariableop_48_batch_normalization_5_moving_varianceIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_5_moving_stddevIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp5assignvariableop_50_batch_normalization_5_renorm_meanIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp7assignvariableop_51_batch_normalization_5_renorm_stddevIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp"assignvariableop_52_output1_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp assignvariableop_53_output1_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*�
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
L__inference_batch_normalization_layer_call_and_return_conditional_losses_822

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1174

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_4_layer_call_fn_4171

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1436p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_1854

input1

input2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:	�

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:
identity

identity_1
��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput1input2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2
*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0
*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinput1:OK
'
_output_shapes
:���������
 
_user_specified_nameinput2
�
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3595

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4119

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_2_layer_call_and_return_conditional_losses_1675

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_2758
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:	�

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:
identity

identity_1
��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2
*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_1777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0
*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1
�
q
E__inference_concatenate_layer_call_and_return_conditional_losses_3543
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1
�Q
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1043

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_3_layer_call_and_return_conditional_losses_1701

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_dense_3_layer_call_fn_4128

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1701p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�P
�
L__inference_batch_normalization_layer_call_and_return_conditional_losses_912

inputs-
maximum_readvariableop_resource:)
sub_readvariableop_resource:)
mul_readvariableop_resource:+
add_1_readvariableop_resource:7
)assignmovingavg_2_readvariableop_resource:7
)assignmovingavg_3_readvariableop_resource:%
assignnewvalue_resource:

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:]
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes
::
SqrtSqrtadd:z:0*
T0*
_output_shapes
:M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: r
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
:*
dtype0c
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes
:v
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:j
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0e
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes
:w
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:i
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes
:[

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes
:N

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes
:Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<v
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0|
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0P

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes
:\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<|
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0t

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes
:}

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes
:J
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes
:L
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes
:j
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:*
dtype0^
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
:l
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:*
dtype0b
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0\
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes
:\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:Z
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes
:>
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes
:\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes
:*
dtype0w
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes
:~
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes
:*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes
:*
dtype0g
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:N
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes
:<
ReluRelu	sub_1:z:0*
T0*
_output_shapes
:�
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:W
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Y
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������d

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_dense_1_layer_call_and_return_conditional_losses_1649

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_3_layer_call_fn_4003

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1215p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_dense_4_layer_call_fn_4277

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1727p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1436

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_2370

input1

input2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:
��

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:
��

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:
��

unknown_43:	�

unknown_44:	�

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	�

unknown_50:	�

unknown_51:	�

unknown_52:
identity

identity_1
��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput1input2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2
*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*:
_read_only_resource_inputs
	
 $%()-.1267*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0
*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinput1:OK
'
_output_shapes
:���������
 
_user_specified_nameinput2
�

�
A__inference_dense_4_layer_call_and_return_conditional_losses_4288

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
o
E__inference_concatenate_layer_call_and_return_conditional_losses_1601

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3744

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�]
�
?__inference_model_layer_call_and_return_conditional_losses_2141

inputs
inputs_1&
batch_normalization_2016:&
batch_normalization_2018:&
batch_normalization_2020:&
batch_normalization_2022:&
batch_normalization_2024:&
batch_normalization_2026:&
batch_normalization_2028:

dense_2031:	�

dense_2033:	�)
batch_normalization_1_2036:	�)
batch_normalization_1_2038:	�)
batch_normalization_1_2040:	�)
batch_normalization_1_2042:	�)
batch_normalization_1_2044:	�)
batch_normalization_1_2046:	�)
batch_normalization_1_2048:	� 
dense_1_2051:
��
dense_1_2053:	�)
batch_normalization_2_2056:	�)
batch_normalization_2_2058:	�)
batch_normalization_2_2060:	�)
batch_normalization_2_2062:	�)
batch_normalization_2_2064:	�)
batch_normalization_2_2066:	�)
batch_normalization_2_2068:	� 
dense_2_2071:
��
dense_2_2073:	�)
batch_normalization_3_2076:	�)
batch_normalization_3_2078:	�)
batch_normalization_3_2080:	�)
batch_normalization_3_2082:	�)
batch_normalization_3_2084:	�)
batch_normalization_3_2086:	�)
batch_normalization_3_2088:	� 
dense_3_2091:
��
dense_3_2093:	�)
batch_normalization_4_2096:	�)
batch_normalization_4_2098:	�)
batch_normalization_4_2100:	�)
batch_normalization_4_2102:	�)
batch_normalization_4_2104:	�)
batch_normalization_4_2106:	�)
batch_normalization_4_2108:	� 
dense_4_2111:
��
dense_4_2113:	�)
batch_normalization_5_2116:	�)
batch_normalization_5_2118:	�)
batch_normalization_5_2120:	�)
batch_normalization_5_2122:	�)
batch_normalization_5_2124:	�)
batch_normalization_5_2126:	�)
batch_normalization_5_2128:	�
output1_2131:	�
output1_2133:
identity

identity_1
��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�output1/StatefulPartitionedCallc
concatenate/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCallinputsconcatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1601�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_2016batch_normalization_2018batch_normalization_2020batch_normalization_2022batch_normalization_2024batch_normalization_2026batch_normalization_2028*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_912�
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
dense_2031
dense_2033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1623�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_2036batch_normalization_1_2038batch_normalization_1_2040batch_normalization_1_2042batch_normalization_1_2044batch_normalization_1_2046batch_normalization_1_2048*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1043�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_2051dense_1_2053*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1649�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_2056batch_normalization_2_2058batch_normalization_2_2060batch_normalization_2_2062batch_normalization_2_2064batch_normalization_2_2066batch_normalization_2_2068*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1174�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_2071dense_2_2073*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1675�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_3_2076batch_normalization_3_2078batch_normalization_3_2080batch_normalization_3_2082batch_normalization_3_2084batch_normalization_3_2086batch_normalization_3_2088*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1305�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_3_2091dense_3_2093*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1701�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_4_2096batch_normalization_4_2098batch_normalization_4_2100batch_normalization_4_2102batch_normalization_4_2104batch_normalization_4_2106batch_normalization_4_2108*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1436�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_4_2111dense_4_2113*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1727�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_5_2116batch_normalization_5_2118batch_normalization_5_2120batch_normalization_5_2122batch_normalization_5_2124batch_normalization_5_2126batch_normalization_5_2128*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1567�
output1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0output1_2131output1_2133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output1_layer_call_and_return_conditional_losses_1753^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater/GreaterGreater(output1/StatefulPartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*'
_output_shapes
:����������
output2/PartitionedCallPartitionedCalltf.math.greater/Greater:z:0*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output2_layer_call_and_return_conditional_losses_1773w
IdentityIdentity(output1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity output2/PartitionedCall:output:0^NoOp*
T0
*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^output1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
]
A__inference_output2_layer_call_and_return_conditional_losses_4454

inputs

identity
I
ShapeShapeinputs*
T0
*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0
*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0
*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�S
�
?__inference_model_layer_call_and_return_conditional_losses_1777

inputs
inputs_1&
batch_normalization_1603:&
batch_normalization_1605:&
batch_normalization_1607:&
batch_normalization_1609:

dense_1624:	�

dense_1626:	�)
batch_normalization_1_1629:	�)
batch_normalization_1_1631:	�)
batch_normalization_1_1633:	�)
batch_normalization_1_1635:	� 
dense_1_1650:
��
dense_1_1652:	�)
batch_normalization_2_1655:	�)
batch_normalization_2_1657:	�)
batch_normalization_2_1659:	�)
batch_normalization_2_1661:	� 
dense_2_1676:
��
dense_2_1678:	�)
batch_normalization_3_1681:	�)
batch_normalization_3_1683:	�)
batch_normalization_3_1685:	�)
batch_normalization_3_1687:	� 
dense_3_1702:
��
dense_3_1704:	�)
batch_normalization_4_1707:	�)
batch_normalization_4_1709:	�)
batch_normalization_4_1711:	�)
batch_normalization_4_1713:	� 
dense_4_1728:
��
dense_4_1730:	�)
batch_normalization_5_1733:	�)
batch_normalization_5_1735:	�)
batch_normalization_5_1737:	�)
batch_normalization_5_1739:	�
output1_1754:	�
output1_1756:
identity

identity_1
��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�output1/StatefulPartitionedCallc
concatenate/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCallinputsconcatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1601�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1603batch_normalization_1605batch_normalization_1607batch_normalization_1609*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_822�
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
dense_1624
dense_1626*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1623�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_1629batch_normalization_1_1631batch_normalization_1_1633batch_normalization_1_1635*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_953�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_1650dense_1_1652*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1649�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_1655batch_normalization_2_1657batch_normalization_2_1659batch_normalization_2_1661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1084�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_1676dense_2_1678*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1675�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_3_1681batch_normalization_3_1683batch_normalization_3_1685batch_normalization_3_1687*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1215�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_3_1702dense_3_1704*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1701�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_4_1707batch_normalization_4_1709batch_normalization_4_1711batch_normalization_4_1713*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1346�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_4_1728dense_4_1730*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1727�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_5_1733batch_normalization_5_1735batch_normalization_5_1737batch_normalization_5_1739*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1477�
output1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0output1_1754output1_1756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output1_layer_call_and_return_conditional_losses_1753^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater/GreaterGreater(output1/StatefulPartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*'
_output_shapes
:����������
output2/PartitionedCallPartitionedCalltf.math.greater/Greater:z:0*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output2_layer_call_and_return_conditional_losses_1773w
IdentityIdentity(output1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity output2/PartitionedCall:output:0^NoOp*
T0
*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^output1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�#
__inference__wrapped_model_798

input1

input2I
;model_batch_normalization_batchnorm_readvariableop_resource:M
?model_batch_normalization_batchnorm_mul_readvariableop_resource:K
=model_batch_normalization_batchnorm_readvariableop_1_resource:K
=model_batch_normalization_batchnorm_readvariableop_2_resource:=
*model_dense_matmul_readvariableop_resource:	�:
+model_dense_biasadd_readvariableop_resource:	�L
=model_batch_normalization_1_batchnorm_readvariableop_resource:	�P
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�N
?model_batch_normalization_1_batchnorm_readvariableop_1_resource:	�N
?model_batch_normalization_1_batchnorm_readvariableop_2_resource:	�@
,model_dense_1_matmul_readvariableop_resource:
��<
-model_dense_1_biasadd_readvariableop_resource:	�L
=model_batch_normalization_2_batchnorm_readvariableop_resource:	�P
Amodel_batch_normalization_2_batchnorm_mul_readvariableop_resource:	�N
?model_batch_normalization_2_batchnorm_readvariableop_1_resource:	�N
?model_batch_normalization_2_batchnorm_readvariableop_2_resource:	�@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�L
=model_batch_normalization_3_batchnorm_readvariableop_resource:	�P
Amodel_batch_normalization_3_batchnorm_mul_readvariableop_resource:	�N
?model_batch_normalization_3_batchnorm_readvariableop_1_resource:	�N
?model_batch_normalization_3_batchnorm_readvariableop_2_resource:	�@
,model_dense_3_matmul_readvariableop_resource:
��<
-model_dense_3_biasadd_readvariableop_resource:	�L
=model_batch_normalization_4_batchnorm_readvariableop_resource:	�P
Amodel_batch_normalization_4_batchnorm_mul_readvariableop_resource:	�N
?model_batch_normalization_4_batchnorm_readvariableop_1_resource:	�N
?model_batch_normalization_4_batchnorm_readvariableop_2_resource:	�@
,model_dense_4_matmul_readvariableop_resource:
��<
-model_dense_4_biasadd_readvariableop_resource:	�L
=model_batch_normalization_5_batchnorm_readvariableop_resource:	�P
Amodel_batch_normalization_5_batchnorm_mul_readvariableop_resource:	�N
?model_batch_normalization_5_batchnorm_readvariableop_1_resource:	�N
?model_batch_normalization_5_batchnorm_readvariableop_2_resource:	�?
,model_output1_matmul_readvariableop_resource:	�;
-model_output1_biasadd_readvariableop_resource:
identity

identity_1
��2model/batch_normalization/batchnorm/ReadVariableOp�4model/batch_normalization/batchnorm/ReadVariableOp_1�4model/batch_normalization/batchnorm/ReadVariableOp_2�6model/batch_normalization/batchnorm/mul/ReadVariableOp�4model/batch_normalization_1/batchnorm/ReadVariableOp�6model/batch_normalization_1/batchnorm/ReadVariableOp_1�6model/batch_normalization_1/batchnorm/ReadVariableOp_2�8model/batch_normalization_1/batchnorm/mul/ReadVariableOp�4model/batch_normalization_2/batchnorm/ReadVariableOp�6model/batch_normalization_2/batchnorm/ReadVariableOp_1�6model/batch_normalization_2/batchnorm/ReadVariableOp_2�8model/batch_normalization_2/batchnorm/mul/ReadVariableOp�4model/batch_normalization_3/batchnorm/ReadVariableOp�6model/batch_normalization_3/batchnorm/ReadVariableOp_1�6model/batch_normalization_3/batchnorm/ReadVariableOp_2�8model/batch_normalization_3/batchnorm/mul/ReadVariableOp�4model/batch_normalization_4/batchnorm/ReadVariableOp�6model/batch_normalization_4/batchnorm/ReadVariableOp_1�6model/batch_normalization_4/batchnorm/ReadVariableOp_2�8model/batch_normalization_4/batchnorm/mul/ReadVariableOp�4model/batch_normalization_5/batchnorm/ReadVariableOp�6model/batch_normalization_5/batchnorm/ReadVariableOp_1�6model/batch_normalization_5/batchnorm/ReadVariableOp_2�8model/batch_normalization_5/batchnorm/mul/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOp�$model/output1/BiasAdd/ReadVariableOp�#model/output1/MatMul/ReadVariableOpg
model/concatenate/CastCastinput2*

DstT0*

SrcT0*'
_output_shapes
:���������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2input1model/concatenate/Cast:y:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
)model/batch_normalization/batchnorm/mul_1Mul!model/concatenate/concat:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
model/dense/TanhTanhmodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/mul_1Mulmodel/dense/Tanh:y:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_1/MatMulMatMul/model/batch_normalization_1/batchnorm/add_1:z:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_1/TanhTanhmodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4model/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0p
+model/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_2/batchnorm/addAddV2<model/batch_normalization_2/batchnorm/ReadVariableOp:value:04model/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_2/batchnorm/RsqrtRsqrt-model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
8model/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_2/batchnorm/mulMul/model/batch_normalization_2/batchnorm/Rsqrt:y:0@model/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
+model/batch_normalization_2/batchnorm/mul_1Mulmodel/dense_1/Tanh:y:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6model/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
+model/batch_normalization_2/batchnorm/mul_2Mul>model/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_2/batchnorm/subSub>model/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
+model/batch_normalization_2/batchnorm/add_1AddV2/model/batch_normalization_2/batchnorm/mul_1:z:0-model/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul/model/batch_normalization_2/batchnorm/add_1:z:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_2/TanhTanhmodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4model/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0p
+model/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_3/batchnorm/addAddV2<model/batch_normalization_3/batchnorm/ReadVariableOp:value:04model/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_3/batchnorm/RsqrtRsqrt-model/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:��
8model/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_3/batchnorm/mulMul/model/batch_normalization_3/batchnorm/Rsqrt:y:0@model/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
+model/batch_normalization_3/batchnorm/mul_1Mulmodel/dense_2/Tanh:y:0-model/batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6model/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
+model/batch_normalization_3/batchnorm/mul_2Mul>model/batch_normalization_3/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_3/batchnorm/subSub>model/batch_normalization_3/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
+model/batch_normalization_3/batchnorm/add_1AddV2/model/batch_normalization_3/batchnorm/mul_1:z:0-model/batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_3/MatMulMatMul/model/batch_normalization_3/batchnorm/add_1:z:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_3/TanhTanhmodel/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4model/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0p
+model/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_4/batchnorm/addAddV2<model/batch_normalization_4/batchnorm/ReadVariableOp:value:04model/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_4/batchnorm/RsqrtRsqrt-model/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:��
8model/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_4/batchnorm/mulMul/model/batch_normalization_4/batchnorm/Rsqrt:y:0@model/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
+model/batch_normalization_4/batchnorm/mul_1Mulmodel/dense_3/Tanh:y:0-model/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6model/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
+model/batch_normalization_4/batchnorm/mul_2Mul>model/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_4/batchnorm/subSub>model/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
+model/batch_normalization_4/batchnorm/add_1AddV2/model/batch_normalization_4/batchnorm/mul_1:z:0-model/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_4/MatMulMatMul/model/batch_normalization_4/batchnorm/add_1:z:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
model/dense_4/TanhTanhmodel/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4model/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0p
+model/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_5/batchnorm/addAddV2<model/batch_normalization_5/batchnorm/ReadVariableOp:value:04model/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_5/batchnorm/RsqrtRsqrt-model/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:��
8model/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_5/batchnorm/mulMul/model/batch_normalization_5/batchnorm/Rsqrt:y:0@model/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
+model/batch_normalization_5/batchnorm/mul_1Mulmodel/dense_4/Tanh:y:0-model/batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6model/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
+model/batch_normalization_5/batchnorm/mul_2Mul>model/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_5/batchnorm/subSub>model/batch_normalization_5/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
+model/batch_normalization_5/batchnorm/add_1AddV2/model/batch_normalization_5/batchnorm/mul_1:z:0-model/batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
#model/output1/MatMul/ReadVariableOpReadVariableOp,model_output1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/output1/MatMulMatMul/model/batch_normalization_5/batchnorm/add_1:z:0+model/output1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/output1/BiasAdd/ReadVariableOpReadVariableOp-model_output1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/output1/BiasAddBiasAddmodel/output1/MatMul:product:0,model/output1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model/output1/SoftmaxSoftmaxmodel/output1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d
model/tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
model/tf.math.greater/GreaterGreatermodel/output1/Softmax:softmax:0(model/tf.math.greater/Greater/y:output:0*
T0*'
_output_shapes
:���������r
model/output2/ShapeShape!model/tf.math.greater/Greater:z:0*
T0
*
_output_shapes
::��k
!model/output2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/output2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/output2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/output2/strided_sliceStridedSlicemodel/output2/Shape:output:0*model/output2/strided_slice/stack:output:0,model/output2/strided_slice/stack_1:output:0,model/output2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/output2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
model/output2/Reshape/shapePack$model/output2/strided_slice:output:0&model/output2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/output2/ReshapeReshape!model/tf.math.greater/Greater:z:0$model/output2/Reshape/shape:output:0*
T0
*'
_output_shapes
:���������n
IdentityIdentitymodel/output1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������o

Identity_1Identitymodel/output2/Reshape:output:0^NoOp*
T0
*'
_output_shapes
:����������
NoOpNoOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp5^model/batch_normalization_2/batchnorm/ReadVariableOp7^model/batch_normalization_2/batchnorm/ReadVariableOp_17^model/batch_normalization_2/batchnorm/ReadVariableOp_29^model/batch_normalization_2/batchnorm/mul/ReadVariableOp5^model/batch_normalization_3/batchnorm/ReadVariableOp7^model/batch_normalization_3/batchnorm/ReadVariableOp_17^model/batch_normalization_3/batchnorm/ReadVariableOp_29^model/batch_normalization_3/batchnorm/mul/ReadVariableOp5^model/batch_normalization_4/batchnorm/ReadVariableOp7^model/batch_normalization_4/batchnorm/ReadVariableOp_17^model/batch_normalization_4/batchnorm/ReadVariableOp_29^model/batch_normalization_4/batchnorm/mul/ReadVariableOp5^model/batch_normalization_5/batchnorm/ReadVariableOp7^model/batch_normalization_5/batchnorm/ReadVariableOp_17^model/batch_normalization_5/batchnorm/ReadVariableOp_29^model/batch_normalization_5/batchnorm/mul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/output1/BiasAdd/ReadVariableOp$^model/output1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_2/batchnorm/ReadVariableOp4model/batch_normalization_2/batchnorm/ReadVariableOp2p
6model/batch_normalization_2/batchnorm/ReadVariableOp_16model/batch_normalization_2/batchnorm/ReadVariableOp_12p
6model/batch_normalization_2/batchnorm/ReadVariableOp_26model/batch_normalization_2/batchnorm/ReadVariableOp_22t
8model/batch_normalization_2/batchnorm/mul/ReadVariableOp8model/batch_normalization_2/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_3/batchnorm/ReadVariableOp4model/batch_normalization_3/batchnorm/ReadVariableOp2p
6model/batch_normalization_3/batchnorm/ReadVariableOp_16model/batch_normalization_3/batchnorm/ReadVariableOp_12p
6model/batch_normalization_3/batchnorm/ReadVariableOp_26model/batch_normalization_3/batchnorm/ReadVariableOp_22t
8model/batch_normalization_3/batchnorm/mul/ReadVariableOp8model/batch_normalization_3/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_4/batchnorm/ReadVariableOp4model/batch_normalization_4/batchnorm/ReadVariableOp2p
6model/batch_normalization_4/batchnorm/ReadVariableOp_16model/batch_normalization_4/batchnorm/ReadVariableOp_12p
6model/batch_normalization_4/batchnorm/ReadVariableOp_26model/batch_normalization_4/batchnorm/ReadVariableOp_22t
8model/batch_normalization_4/batchnorm/mul/ReadVariableOp8model/batch_normalization_4/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_5/batchnorm/ReadVariableOp4model/batch_normalization_5/batchnorm/ReadVariableOp2p
6model/batch_normalization_5/batchnorm/ReadVariableOp_16model/batch_normalization_5/batchnorm/ReadVariableOp_12p
6model/batch_normalization_5/batchnorm/ReadVariableOp_26model/batch_normalization_5/batchnorm/ReadVariableOp_22t
8model/batch_normalization_5/batchnorm/mul/ReadVariableOp8model/batch_normalization_5/batchnorm/mul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/output1/BiasAdd/ReadVariableOp$model/output1/BiasAdd/ReadVariableOp2J
#model/output1/MatMul/ReadVariableOp#model/output1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinput1:OK
'
_output_shapes
:���������
 
_user_specified_nameinput2
�S
�
?__inference_model_layer_call_and_return_conditional_losses_2465

input1

input2&
batch_normalization_2376:&
batch_normalization_2378:&
batch_normalization_2380:&
batch_normalization_2382:

dense_2385:	�

dense_2387:	�)
batch_normalization_1_2390:	�)
batch_normalization_1_2392:	�)
batch_normalization_1_2394:	�)
batch_normalization_1_2396:	� 
dense_1_2399:
��
dense_1_2401:	�)
batch_normalization_2_2404:	�)
batch_normalization_2_2406:	�)
batch_normalization_2_2408:	�)
batch_normalization_2_2410:	� 
dense_2_2413:
��
dense_2_2415:	�)
batch_normalization_3_2418:	�)
batch_normalization_3_2420:	�)
batch_normalization_3_2422:	�)
batch_normalization_3_2424:	� 
dense_3_2427:
��
dense_3_2429:	�)
batch_normalization_4_2432:	�)
batch_normalization_4_2434:	�)
batch_normalization_4_2436:	�)
batch_normalization_4_2438:	� 
dense_4_2441:
��
dense_4_2443:	�)
batch_normalization_5_2446:	�)
batch_normalization_5_2448:	�)
batch_normalization_5_2450:	�)
batch_normalization_5_2452:	�
output1_2455:	�
output1_2457:
identity

identity_1
��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�output1/StatefulPartitionedCalla
concatenate/CastCastinput2*

DstT0*

SrcT0*'
_output_shapes
:����������
concatenate/PartitionedCallPartitionedCallinput1concatenate/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1601�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_2376batch_normalization_2378batch_normalization_2380batch_normalization_2382*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_822�
dense/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
dense_2385
dense_2387*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1623�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_1_2390batch_normalization_1_2392batch_normalization_1_2394batch_normalization_1_2396*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_953�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_1_2399dense_1_2401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1649�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_2404batch_normalization_2_2406batch_normalization_2_2408batch_normalization_2_2410*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1084�
dense_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0dense_2_2413dense_2_2415*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_1675�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_3_2418batch_normalization_3_2420batch_normalization_3_2422batch_normalization_3_2424*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1215�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0dense_3_2427dense_3_2429*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_1701�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0batch_normalization_4_2432batch_normalization_4_2434batch_normalization_4_2436batch_normalization_4_2438*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1346�
dense_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0dense_4_2441dense_4_2443*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_1727�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0batch_normalization_5_2446batch_normalization_5_2448batch_normalization_5_2450batch_normalization_5_2452*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1477�
output1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0output1_2455output1_2457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output1_layer_call_and_return_conditional_losses_1753^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater/GreaterGreater(output1/StatefulPartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*'
_output_shapes
:����������
output2/PartitionedCallPartitionedCalltf.math.greater/Greater:z:0*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output2_layer_call_and_return_conditional_losses_1773w
IdentityIdentity(output1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity output2/PartitionedCall:output:0^NoOp*
T0
*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^output1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
output1/StatefulPartitionedCalloutput1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinput1:OK
'
_output_shapes
:���������
 
_user_specified_nameinput2
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3893

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_2_layer_call_fn_3854

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1084p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_1_layer_call_fn_3705

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_953p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_dense_1_layer_call_fn_3830

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_1649p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_output1_layer_call_and_return_conditional_losses_4437

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_3_layer_call_and_return_conditional_losses_4139

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4042

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4340

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_3_layer_call_fn_4022

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1305p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_output2_layer_call_fn_4442

inputs

identity
�
PartitionedCallPartitionedCallinputs*
Tin
2
*
Tout
2
*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output2_layer_call_and_return_conditional_losses_1773`
IdentityIdentityPartitionedCall:output:0*
T0
*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3970

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_2_layer_call_and_return_conditional_losses_3990

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_dense_layer_call_fn_3681

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1623p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
4__inference_batch_normalization_5_layer_call_fn_4320

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1567p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
2__inference_batch_normalization_layer_call_fn_3575

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_2874
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:
��

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:
��

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:
��

unknown_43:	�

unknown_44:	�

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	�

unknown_50:	�

unknown_51:	�

unknown_52:
identity

identity_1
��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2
*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*:
_read_only_resource_inputs
	
 $%()-.1267*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0
*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1
�Q
�
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1567

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3821

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_output1_layer_call_and_return_conditional_losses_1753

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
2__inference_batch_normalization_layer_call_fn_3556

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_822o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1215

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Q
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1305

inputs.
maximum_readvariableop_resource:	�*
sub_readvariableop_resource:	�*
mul_readvariableop_resource:	�,
add_1_readvariableop_resource:	�8
)assignmovingavg_2_readvariableop_resource:	�8
)assignmovingavg_3_readvariableop_resource:	�&
assignnewvalue_resource:	�

identity_6��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�AssignMovingAvg_2� AssignMovingAvg_2/ReadVariableOp�AssignMovingAvg_3� AssignMovingAvg_3/ReadVariableOp�AssignNewValue�Maximum/ReadVariableOp�ReadVariableOp�add_1/ReadVariableOp�mul/ReadVariableOp�mul_1/ReadVariableOp�mul_2/ReadVariableOp�sub/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:^
addAddV2moments/Squeeze_1:output:0add/y:output:0*
T0*
_output_shapes	
:�;
SqrtSqrtadd:z:0*
T0*
_output_shapes	
:�M
Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:B
Sqrt_1SqrtSqrt_1/x:output:0*
T0*
_output_shapes
: s
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0d
MaximumMaximumMaximum/ReadVariableOp:value:0
Sqrt_1:y:0*
T0*
_output_shapes	
:�w
truedivRealDivSqrt:y:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�k
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0f
subSubmoments/Squeeze:output:0sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:�x
	truediv_1RealDivsub:z:0Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:�j
IdentityIdentitymoments/Squeeze:output:0^truediv
^truediv_1*
T0*
_output_shapes	
:�\

Identity_1IdentitySqrt:y:0^truediv
^truediv_1*
T0*
_output_shapes	
:�O

Identity_2IdentityIdentity:output:0*
T0*
_output_shapes	
:�Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<w
AssignMovingAvg/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes	
:�*
dtype0}
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0Identity_2:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOpsub_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp^sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Q

Identity_3IdentityIdentity_1:output:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<}
 AssignMovingAvg_1/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0Identity_3:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOpmaximum_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp^Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0u

Identity_4IdentityIdentity:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�~

Identity_5Identitymoments/Squeeze_1:output:0^AssignMovingAvg^AssignMovingAvg_1*
T0*
_output_shapes	
:�K
renorm_rStopGradienttruediv:z:0*
T0*
_output_shapes	
:�M
renorm_dStopGradienttruediv_1:z:0*
T0*
_output_shapes	
:�k
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0_
mulMulrenorm_r:output:0mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�m
mul_1/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes	
:�*
dtype0c
mul_1Mulrenorm_d:output:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�o
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes	
:�*
dtype0]
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_2/ReadVariableOpReadVariableOp)assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_2/subSub(AssignMovingAvg_2/ReadVariableOp:value:0Identity_4:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_2/mulMulAssignMovingAvg_2/sub:z:0 AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_2AssignSubVariableOp)assignmovingavg_2_readvariableop_resourceAssignMovingAvg_2/mul:z:0!^AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:[
add_2AddV2Identity_5:output:0add_2/y:output:0*
T0*
_output_shapes	
:�?
Sqrt_2Sqrt	add_2:z:0*
T0*
_output_shapes	
:�\
AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_3/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0x
AssignMovingAvg_3/subSub(AssignMovingAvg_3/ReadVariableOp:value:0
Sqrt_2:y:0*
T0*
_output_shapes	
:�
AssignMovingAvg_3/mulMulAssignMovingAvg_3/sub:z:0 AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_3AssignSubVariableOp)assignmovingavg_3_readvariableop_resourceAssignMovingAvg_3/mul:z:0!^AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
mul_2/ReadVariableOpReadVariableOp)assignmovingavg_3_readvariableop_resource^AssignMovingAvg_3*
_output_shapes	
:�*
dtype0h
mul_2MulReadVariableOp:value:0mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:O
sub_1Sub	mul_2:z:0sub_1/y:output:0*
T0*
_output_shapes	
:�=
ReluRelu	sub_1:z:0*
T0*
_output_shapes	
:��
AssignNewValueAssignVariableOpassignnewvalue_resourceRelu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�X
batchnorm/mulMulbatchnorm/Rsqrt:y:0mul:z:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�Z
batchnorm/subSub	add_1:z:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������e

Identity_6Identitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^AssignMovingAvg_2!^AssignMovingAvg_2/ReadVariableOp^AssignMovingAvg_3!^AssignMovingAvg_3/ReadVariableOp^AssignNewValue^Maximum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp^mul/ReadVariableOp^mul_1/ReadVariableOp^mul_2/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_2AssignMovingAvg_22D
 AssignMovingAvg_2/ReadVariableOp AssignMovingAvg_2/ReadVariableOp2&
AssignMovingAvg_3AssignMovingAvg_32D
 AssignMovingAvg_3/ReadVariableOp AssignMovingAvg_3/ReadVariableOp2 
AssignNewValueAssignNewValue20
Maximum/ReadVariableOpMaximum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp2,
mul_1/ReadVariableOpmul_1/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
ɒ
�B
?__inference_model_layer_call_and_return_conditional_losses_3530
inputs_0
inputs_1A
3batch_normalization_maximum_readvariableop_resource:=
/batch_normalization_sub_readvariableop_resource:=
/batch_normalization_mul_readvariableop_resource:?
1batch_normalization_add_1_readvariableop_resource:K
=batch_normalization_assignmovingavg_2_readvariableop_resource:K
=batch_normalization_assignmovingavg_3_readvariableop_resource:9
+batch_normalization_assignnewvalue_resource:7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�D
5batch_normalization_1_maximum_readvariableop_resource:	�@
1batch_normalization_1_sub_readvariableop_resource:	�@
1batch_normalization_1_mul_readvariableop_resource:	�B
3batch_normalization_1_add_1_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_2_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_3_readvariableop_resource:	�<
-batch_normalization_1_assignnewvalue_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�D
5batch_normalization_2_maximum_readvariableop_resource:	�@
1batch_normalization_2_sub_readvariableop_resource:	�@
1batch_normalization_2_mul_readvariableop_resource:	�B
3batch_normalization_2_add_1_readvariableop_resource:	�N
?batch_normalization_2_assignmovingavg_2_readvariableop_resource:	�N
?batch_normalization_2_assignmovingavg_3_readvariableop_resource:	�<
-batch_normalization_2_assignnewvalue_resource:	�:
&dense_2_matmul_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�D
5batch_normalization_3_maximum_readvariableop_resource:	�@
1batch_normalization_3_sub_readvariableop_resource:	�@
1batch_normalization_3_mul_readvariableop_resource:	�B
3batch_normalization_3_add_1_readvariableop_resource:	�N
?batch_normalization_3_assignmovingavg_2_readvariableop_resource:	�N
?batch_normalization_3_assignmovingavg_3_readvariableop_resource:	�<
-batch_normalization_3_assignnewvalue_resource:	�:
&dense_3_matmul_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�D
5batch_normalization_4_maximum_readvariableop_resource:	�@
1batch_normalization_4_sub_readvariableop_resource:	�@
1batch_normalization_4_mul_readvariableop_resource:	�B
3batch_normalization_4_add_1_readvariableop_resource:	�N
?batch_normalization_4_assignmovingavg_2_readvariableop_resource:	�N
?batch_normalization_4_assignmovingavg_3_readvariableop_resource:	�<
-batch_normalization_4_assignnewvalue_resource:	�:
&dense_4_matmul_readvariableop_resource:
��6
'dense_4_biasadd_readvariableop_resource:	�D
5batch_normalization_5_maximum_readvariableop_resource:	�@
1batch_normalization_5_sub_readvariableop_resource:	�@
1batch_normalization_5_mul_readvariableop_resource:	�B
3batch_normalization_5_add_1_readvariableop_resource:	�N
?batch_normalization_5_assignmovingavg_2_readvariableop_resource:	�N
?batch_normalization_5_assignmovingavg_3_readvariableop_resource:	�<
-batch_normalization_5_assignnewvalue_resource:	�9
&output1_matmul_readvariableop_resource:	�5
'output1_biasadd_readvariableop_resource:
identity

identity_1
��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�%batch_normalization/AssignMovingAvg_2�4batch_normalization/AssignMovingAvg_2/ReadVariableOp�%batch_normalization/AssignMovingAvg_3�4batch_normalization/AssignMovingAvg_3/ReadVariableOp�"batch_normalization/AssignNewValue�*batch_normalization/Maximum/ReadVariableOp�"batch_normalization/ReadVariableOp�(batch_normalization/add_1/ReadVariableOp�&batch_normalization/mul/ReadVariableOp�(batch_normalization/mul_1/ReadVariableOp�(batch_normalization/mul_2/ReadVariableOp�&batch_normalization/sub/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_2�6batch_normalization_1/AssignMovingAvg_2/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_3�6batch_normalization_1/AssignMovingAvg_3/ReadVariableOp�$batch_normalization_1/AssignNewValue�,batch_normalization_1/Maximum/ReadVariableOp�$batch_normalization_1/ReadVariableOp�*batch_normalization_1/add_1/ReadVariableOp�(batch_normalization_1/mul/ReadVariableOp�*batch_normalization_1/mul_1/ReadVariableOp�*batch_normalization_1/mul_2/ReadVariableOp�(batch_normalization_1/sub/ReadVariableOp�%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_2�6batch_normalization_2/AssignMovingAvg_2/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_3�6batch_normalization_2/AssignMovingAvg_3/ReadVariableOp�$batch_normalization_2/AssignNewValue�,batch_normalization_2/Maximum/ReadVariableOp�$batch_normalization_2/ReadVariableOp�*batch_normalization_2/add_1/ReadVariableOp�(batch_normalization_2/mul/ReadVariableOp�*batch_normalization_2/mul_1/ReadVariableOp�*batch_normalization_2/mul_2/ReadVariableOp�(batch_normalization_2/sub/ReadVariableOp�%batch_normalization_3/AssignMovingAvg�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_1�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_2�6batch_normalization_3/AssignMovingAvg_2/ReadVariableOp�'batch_normalization_3/AssignMovingAvg_3�6batch_normalization_3/AssignMovingAvg_3/ReadVariableOp�$batch_normalization_3/AssignNewValue�,batch_normalization_3/Maximum/ReadVariableOp�$batch_normalization_3/ReadVariableOp�*batch_normalization_3/add_1/ReadVariableOp�(batch_normalization_3/mul/ReadVariableOp�*batch_normalization_3/mul_1/ReadVariableOp�*batch_normalization_3/mul_2/ReadVariableOp�(batch_normalization_3/sub/ReadVariableOp�%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_2�6batch_normalization_4/AssignMovingAvg_2/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_3�6batch_normalization_4/AssignMovingAvg_3/ReadVariableOp�$batch_normalization_4/AssignNewValue�,batch_normalization_4/Maximum/ReadVariableOp�$batch_normalization_4/ReadVariableOp�*batch_normalization_4/add_1/ReadVariableOp�(batch_normalization_4/mul/ReadVariableOp�*batch_normalization_4/mul_1/ReadVariableOp�*batch_normalization_4/mul_2/ReadVariableOp�(batch_normalization_4/sub/ReadVariableOp�%batch_normalization_5/AssignMovingAvg�4batch_normalization_5/AssignMovingAvg/ReadVariableOp�'batch_normalization_5/AssignMovingAvg_1�6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�'batch_normalization_5/AssignMovingAvg_2�6batch_normalization_5/AssignMovingAvg_2/ReadVariableOp�'batch_normalization_5/AssignMovingAvg_3�6batch_normalization_5/AssignMovingAvg_3/ReadVariableOp�$batch_normalization_5/AssignNewValue�,batch_normalization_5/Maximum/ReadVariableOp�$batch_normalization_5/ReadVariableOp�*batch_normalization_5/add_1/ReadVariableOp�(batch_normalization_5/mul/ReadVariableOp�*batch_normalization_5/mul_1/ReadVariableOp�*batch_normalization_5/mul_2/ReadVariableOp�(batch_normalization_5/sub/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�output1/BiasAdd/ReadVariableOp�output1/MatMul/ReadVariableOpc
concatenate/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2inputs_0concatenate/Cast:y:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
 batch_normalization/moments/meanMeanconcatenate/concat:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:�
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconcatenate/concat:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ^
batch_normalization/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization/addAddV2.batch_normalization/moments/Squeeze_1:output:0"batch_normalization/add/y:output:0*
T0*
_output_shapes
:b
batch_normalization/SqrtSqrtbatch_normalization/add:z:0*
T0*
_output_shapes
:a
batch_normalization/Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:j
batch_normalization/Sqrt_1Sqrt%batch_normalization/Sqrt_1/x:output:0*
T0*
_output_shapes
: �
*batch_normalization/Maximum/ReadVariableOpReadVariableOp3batch_normalization_maximum_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_normalization/MaximumMaximum2batch_normalization/Maximum/ReadVariableOp:value:0batch_normalization/Sqrt_1:y:0*
T0*
_output_shapes
:�
batch_normalization/truedivRealDivbatch_normalization/Sqrt:y:0batch_normalization/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:�
&batch_normalization/sub/ReadVariableOpReadVariableOp/batch_normalization_sub_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_normalization/subSub,batch_normalization/moments/Squeeze:output:0.batch_normalization/sub/ReadVariableOp:value:0*
T0*
_output_shapes
:�
batch_normalization/truediv_1RealDivbatch_normalization/sub:z:0batch_normalization/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:�
batch_normalization/IdentityIdentity,batch_normalization/moments/Squeeze:output:0^batch_normalization/truediv^batch_normalization/truediv_1*
T0*
_output_shapes
:�
batch_normalization/Identity_1Identitybatch_normalization/Sqrt:y:0^batch_normalization/truediv^batch_normalization/truediv_1*
T0*
_output_shapes
:v
batch_normalization/Identity_2Identity%batch_normalization/Identity:output:0*
T0*
_output_shapes
:n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_sub_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0'batch_normalization/Identity_2:output:0*
T0*
_output_shapes
:�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
#batch_normalization/AssignMovingAvgAssignSubVariableOp/batch_normalization_sub_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp'^batch_normalization/sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0x
batch_normalization/Identity_3Identity'batch_normalization/Identity_1:output:0*
T0*
_output_shapes
:p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp3batch_normalization_maximum_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0'batch_normalization/Identity_3:output:0*
T0*
_output_shapes
:�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp3batch_normalization_maximum_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp+^batch_normalization/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
batch_normalization/Identity_4Identity%batch_normalization/Identity:output:0$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1*
T0*
_output_shapes
:�
batch_normalization/Identity_5Identity.batch_normalization/moments/Squeeze_1:output:0$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1*
T0*
_output_shapes
:r
batch_normalization/renorm_rStopGradientbatch_normalization/truediv:z:0*
T0*
_output_shapes
:t
batch_normalization/renorm_dStopGradient!batch_normalization/truediv_1:z:0*
T0*
_output_shapes
:�
&batch_normalization/mul/ReadVariableOpReadVariableOp/batch_normalization_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_normalization/mulMul%batch_normalization/renorm_r:output:0.batch_normalization/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization/mul_1/ReadVariableOpReadVariableOp/batch_normalization_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_normalization/mul_1Mul%batch_normalization/renorm_d:output:00batch_normalization/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(batch_normalization/add_1/ReadVariableOpReadVariableOp1batch_normalization_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
batch_normalization/add_1AddV2batch_normalization/mul_1:z:00batch_normalization/add_1/ReadVariableOp:value:0*
T0*
_output_shapes
:p
+batch_normalization/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_2/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_2_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/AssignMovingAvg_2/subSub<batch_normalization/AssignMovingAvg_2/ReadVariableOp:value:0'batch_normalization/Identity_4:output:0*
T0*
_output_shapes
:�
)batch_normalization/AssignMovingAvg_2/mulMul-batch_normalization/AssignMovingAvg_2/sub:z:04batch_normalization/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization/AssignMovingAvg_2AssignSubVariableOp=batch_normalization_assignmovingavg_2_readvariableop_resource-batch_normalization/AssignMovingAvg_2/mul:z:05^batch_normalization/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0`
batch_normalization/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization/add_2AddV2'batch_normalization/Identity_5:output:0$batch_normalization/add_2/y:output:0*
T0*
_output_shapes
:f
batch_normalization/Sqrt_2Sqrtbatch_normalization/add_2:z:0*
T0*
_output_shapes
:p
+batch_normalization/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_3/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_3_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/AssignMovingAvg_3/subSub<batch_normalization/AssignMovingAvg_3/ReadVariableOp:value:0batch_normalization/Sqrt_2:y:0*
T0*
_output_shapes
:�
)batch_normalization/AssignMovingAvg_3/mulMul-batch_normalization/AssignMovingAvg_3/sub:z:04batch_normalization/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization/AssignMovingAvg_3AssignSubVariableOp=batch_normalization_assignmovingavg_3_readvariableop_resource-batch_normalization/AssignMovingAvg_3/mul:z:05^batch_normalization/AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
"batch_normalization/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_3_readvariableop_resource&^batch_normalization/AssignMovingAvg_3*
_output_shapes
:*
dtype0�
(batch_normalization/mul_2/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_3_readvariableop_resource&^batch_normalization/AssignMovingAvg_3*
_output_shapes
:*
dtype0�
batch_normalization/mul_2Mul*batch_normalization/ReadVariableOp:value:00batch_normalization/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes
:`
batch_normalization/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization/sub_1Subbatch_normalization/mul_2:z:0$batch_normalization/sub_1/y:output:0*
T0*
_output_shapes
:d
batch_normalization/ReluRelubatch_normalization/sub_1:z:0*
T0*
_output_shapes
:�
"batch_normalization/AssignNewValueAssignVariableOp+batch_normalization_assignnewvalue_resource&batch_normalization/Relu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:0batch_normalization/mul:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Mulconcatenate/concat:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
!batch_normalization/batchnorm/subSubbatch_normalization/add_1:z:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_1/moments/meanMeandense/Tanh:y:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense/Tanh:y:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 `
batch_normalization_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_1/addAddV20batch_normalization_1/moments/Squeeze_1:output:0$batch_normalization_1/add/y:output:0*
T0*
_output_shapes	
:�g
batch_normalization_1/SqrtSqrtbatch_normalization_1/add:z:0*
T0*
_output_shapes	
:�c
batch_normalization_1/Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:n
batch_normalization_1/Sqrt_1Sqrt'batch_normalization_1/Sqrt_1/x:output:0*
T0*
_output_shapes
: �
,batch_normalization_1/Maximum/ReadVariableOpReadVariableOp5batch_normalization_1_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_1/MaximumMaximum4batch_normalization_1/Maximum/ReadVariableOp:value:0 batch_normalization_1/Sqrt_1:y:0*
T0*
_output_shapes	
:��
batch_normalization_1/truedivRealDivbatch_normalization_1/Sqrt:y:0!batch_normalization_1/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
(batch_normalization_1/sub/ReadVariableOpReadVariableOp1batch_normalization_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_1/subSub.batch_normalization_1/moments/Squeeze:output:00batch_normalization_1/sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
batch_normalization_1/truediv_1RealDivbatch_normalization_1/sub:z:0!batch_normalization_1/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
batch_normalization_1/IdentityIdentity.batch_normalization_1/moments/Squeeze:output:0^batch_normalization_1/truediv ^batch_normalization_1/truediv_1*
T0*
_output_shapes	
:��
 batch_normalization_1/Identity_1Identitybatch_normalization_1/Sqrt:y:0^batch_normalization_1/truediv ^batch_normalization_1/truediv_1*
T0*
_output_shapes	
:�{
 batch_normalization_1/Identity_2Identity'batch_normalization_1/Identity:output:0*
T0*
_output_shapes	
:�p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp1batch_normalization_1_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0)batch_normalization_1/Identity_2:output:0*
T0*
_output_shapes	
:��
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp1batch_normalization_1_sub_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp)^batch_normalization_1/sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0}
 batch_normalization_1/Identity_3Identity)batch_normalization_1/Identity_1:output:0*
T0*
_output_shapes	
:�r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp5batch_normalization_1_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0)batch_normalization_1/Identity_3:output:0*
T0*
_output_shapes	
:��
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp5batch_normalization_1_maximum_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp-^batch_normalization_1/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
 batch_normalization_1/Identity_4Identity'batch_normalization_1/Identity:output:0&^batch_normalization_1/AssignMovingAvg(^batch_normalization_1/AssignMovingAvg_1*
T0*
_output_shapes	
:��
 batch_normalization_1/Identity_5Identity0batch_normalization_1/moments/Squeeze_1:output:0&^batch_normalization_1/AssignMovingAvg(^batch_normalization_1/AssignMovingAvg_1*
T0*
_output_shapes	
:�w
batch_normalization_1/renorm_rStopGradient!batch_normalization_1/truediv:z:0*
T0*
_output_shapes	
:�y
batch_normalization_1/renorm_dStopGradient#batch_normalization_1/truediv_1:z:0*
T0*
_output_shapes	
:��
(batch_normalization_1/mul/ReadVariableOpReadVariableOp1batch_normalization_1_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_1/mulMul'batch_normalization_1/renorm_r:output:00batch_normalization_1/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_1/mul_1/ReadVariableOpReadVariableOp1batch_normalization_1_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_1/mul_1Mul'batch_normalization_1/renorm_d:output:02batch_normalization_1/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_1/add_1/ReadVariableOpReadVariableOp3batch_normalization_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_1/add_1AddV2batch_normalization_1/mul_1:z:02batch_normalization_1/add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�r
-batch_normalization_1/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_1/AssignMovingAvg_2/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_1/AssignMovingAvg_2/subSub>batch_normalization_1/AssignMovingAvg_2/ReadVariableOp:value:0)batch_normalization_1/Identity_4:output:0*
T0*
_output_shapes	
:��
+batch_normalization_1/AssignMovingAvg_2/mulMul/batch_normalization_1/AssignMovingAvg_2/sub:z:06batch_normalization_1/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_1/AssignMovingAvg_2AssignSubVariableOp?batch_normalization_1_assignmovingavg_2_readvariableop_resource/batch_normalization_1/AssignMovingAvg_2/mul:z:07^batch_normalization_1/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0b
batch_normalization_1/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_1/add_2AddV2)batch_normalization_1/Identity_5:output:0&batch_normalization_1/add_2/y:output:0*
T0*
_output_shapes	
:�k
batch_normalization_1/Sqrt_2Sqrtbatch_normalization_1/add_2:z:0*
T0*
_output_shapes	
:�r
-batch_normalization_1/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_1/AssignMovingAvg_3/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_1/AssignMovingAvg_3/subSub>batch_normalization_1/AssignMovingAvg_3/ReadVariableOp:value:0 batch_normalization_1/Sqrt_2:y:0*
T0*
_output_shapes	
:��
+batch_normalization_1/AssignMovingAvg_3/mulMul/batch_normalization_1/AssignMovingAvg_3/sub:z:06batch_normalization_1/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_1/AssignMovingAvg_3AssignSubVariableOp?batch_normalization_1_assignmovingavg_3_readvariableop_resource/batch_normalization_1/AssignMovingAvg_3/mul:z:07^batch_normalization_1/AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
$batch_normalization_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_3_readvariableop_resource(^batch_normalization_1/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
*batch_normalization_1/mul_2/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_3_readvariableop_resource(^batch_normalization_1/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
batch_normalization_1/mul_2Mul,batch_normalization_1/ReadVariableOp:value:02batch_normalization_1/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
batch_normalization_1/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_1/sub_1Subbatch_normalization_1/mul_2:z:0&batch_normalization_1/sub_1/y:output:0*
T0*
_output_shapes	
:�i
batch_normalization_1/ReluRelubatch_normalization_1/sub_1:z:0*
T0*
_output_shapes	
:��
$batch_normalization_1/AssignNewValueAssignVariableOp-batch_normalization_1_assignnewvalue_resource(batch_normalization_1/Relu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0batch_normalization_1/mul:z:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/mul_1Muldense/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_1/batchnorm/subSubbatch_normalization_1/add_1:z:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_2/moments/meanMeandense_1/Tanh:y:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/Tanh:y:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 `
batch_normalization_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_2/addAddV20batch_normalization_2/moments/Squeeze_1:output:0$batch_normalization_2/add/y:output:0*
T0*
_output_shapes	
:�g
batch_normalization_2/SqrtSqrtbatch_normalization_2/add:z:0*
T0*
_output_shapes	
:�c
batch_normalization_2/Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:n
batch_normalization_2/Sqrt_1Sqrt'batch_normalization_2/Sqrt_1/x:output:0*
T0*
_output_shapes
: �
,batch_normalization_2/Maximum/ReadVariableOpReadVariableOp5batch_normalization_2_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_2/MaximumMaximum4batch_normalization_2/Maximum/ReadVariableOp:value:0 batch_normalization_2/Sqrt_1:y:0*
T0*
_output_shapes	
:��
batch_normalization_2/truedivRealDivbatch_normalization_2/Sqrt:y:0!batch_normalization_2/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
(batch_normalization_2/sub/ReadVariableOpReadVariableOp1batch_normalization_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_2/subSub.batch_normalization_2/moments/Squeeze:output:00batch_normalization_2/sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
batch_normalization_2/truediv_1RealDivbatch_normalization_2/sub:z:0!batch_normalization_2/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
batch_normalization_2/IdentityIdentity.batch_normalization_2/moments/Squeeze:output:0^batch_normalization_2/truediv ^batch_normalization_2/truediv_1*
T0*
_output_shapes	
:��
 batch_normalization_2/Identity_1Identitybatch_normalization_2/Sqrt:y:0^batch_normalization_2/truediv ^batch_normalization_2/truediv_1*
T0*
_output_shapes	
:�{
 batch_normalization_2/Identity_2Identity'batch_normalization_2/Identity:output:0*
T0*
_output_shapes	
:�p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp1batch_normalization_2_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0)batch_normalization_2/Identity_2:output:0*
T0*
_output_shapes	
:��
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp1batch_normalization_2_sub_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp)^batch_normalization_2/sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0}
 batch_normalization_2/Identity_3Identity)batch_normalization_2/Identity_1:output:0*
T0*
_output_shapes	
:�r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp5batch_normalization_2_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0)batch_normalization_2/Identity_3:output:0*
T0*
_output_shapes	
:��
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp5batch_normalization_2_maximum_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp-^batch_normalization_2/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
 batch_normalization_2/Identity_4Identity'batch_normalization_2/Identity:output:0&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1*
T0*
_output_shapes	
:��
 batch_normalization_2/Identity_5Identity0batch_normalization_2/moments/Squeeze_1:output:0&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1*
T0*
_output_shapes	
:�w
batch_normalization_2/renorm_rStopGradient!batch_normalization_2/truediv:z:0*
T0*
_output_shapes	
:�y
batch_normalization_2/renorm_dStopGradient#batch_normalization_2/truediv_1:z:0*
T0*
_output_shapes	
:��
(batch_normalization_2/mul/ReadVariableOpReadVariableOp1batch_normalization_2_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_2/mulMul'batch_normalization_2/renorm_r:output:00batch_normalization_2/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_2/mul_1/ReadVariableOpReadVariableOp1batch_normalization_2_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_2/mul_1Mul'batch_normalization_2/renorm_d:output:02batch_normalization_2/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_2/add_1/ReadVariableOpReadVariableOp3batch_normalization_2_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_2/add_1AddV2batch_normalization_2/mul_1:z:02batch_normalization_2/add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�r
-batch_normalization_2/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_2/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_2/AssignMovingAvg_2/subSub>batch_normalization_2/AssignMovingAvg_2/ReadVariableOp:value:0)batch_normalization_2/Identity_4:output:0*
T0*
_output_shapes	
:��
+batch_normalization_2/AssignMovingAvg_2/mulMul/batch_normalization_2/AssignMovingAvg_2/sub:z:06batch_normalization_2/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_2/AssignMovingAvg_2AssignSubVariableOp?batch_normalization_2_assignmovingavg_2_readvariableop_resource/batch_normalization_2/AssignMovingAvg_2/mul:z:07^batch_normalization_2/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0b
batch_normalization_2/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_2/add_2AddV2)batch_normalization_2/Identity_5:output:0&batch_normalization_2/add_2/y:output:0*
T0*
_output_shapes	
:�k
batch_normalization_2/Sqrt_2Sqrtbatch_normalization_2/add_2:z:0*
T0*
_output_shapes	
:�r
-batch_normalization_2/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_3/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_2/AssignMovingAvg_3/subSub>batch_normalization_2/AssignMovingAvg_3/ReadVariableOp:value:0 batch_normalization_2/Sqrt_2:y:0*
T0*
_output_shapes	
:��
+batch_normalization_2/AssignMovingAvg_3/mulMul/batch_normalization_2/AssignMovingAvg_3/sub:z:06batch_normalization_2/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_2/AssignMovingAvg_3AssignSubVariableOp?batch_normalization_2_assignmovingavg_3_readvariableop_resource/batch_normalization_2/AssignMovingAvg_3/mul:z:07^batch_normalization_2/AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
$batch_normalization_2/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_3_readvariableop_resource(^batch_normalization_2/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
*batch_normalization_2/mul_2/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_3_readvariableop_resource(^batch_normalization_2/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
batch_normalization_2/mul_2Mul,batch_normalization_2/ReadVariableOp:value:02batch_normalization_2/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
batch_normalization_2/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_2/sub_1Subbatch_normalization_2/mul_2:z:0&batch_normalization_2/sub_1/y:output:0*
T0*
_output_shapes	
:�i
batch_normalization_2/ReluRelubatch_normalization_2/sub_1:z:0*
T0*
_output_shapes	
:��
$batch_normalization_2/AssignNewValueAssignVariableOp-batch_normalization_2_assignnewvalue_resource(batch_normalization_2/Relu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0batch_normalization_2/mul:z:0*
T0*
_output_shapes	
:��
%batch_normalization_2/batchnorm/mul_1Muldense_1/Tanh:y:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_2/batchnorm/subSubbatch_normalization_2/add_1:z:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_3/moments/meanMeandense_2/Tanh:y:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedense_2/Tanh:y:03batch_normalization_3/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 `
batch_normalization_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_3/addAddV20batch_normalization_3/moments/Squeeze_1:output:0$batch_normalization_3/add/y:output:0*
T0*
_output_shapes	
:�g
batch_normalization_3/SqrtSqrtbatch_normalization_3/add:z:0*
T0*
_output_shapes	
:�c
batch_normalization_3/Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:n
batch_normalization_3/Sqrt_1Sqrt'batch_normalization_3/Sqrt_1/x:output:0*
T0*
_output_shapes
: �
,batch_normalization_3/Maximum/ReadVariableOpReadVariableOp5batch_normalization_3_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_3/MaximumMaximum4batch_normalization_3/Maximum/ReadVariableOp:value:0 batch_normalization_3/Sqrt_1:y:0*
T0*
_output_shapes	
:��
batch_normalization_3/truedivRealDivbatch_normalization_3/Sqrt:y:0!batch_normalization_3/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
(batch_normalization_3/sub/ReadVariableOpReadVariableOp1batch_normalization_3_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_3/subSub.batch_normalization_3/moments/Squeeze:output:00batch_normalization_3/sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
batch_normalization_3/truediv_1RealDivbatch_normalization_3/sub:z:0!batch_normalization_3/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
batch_normalization_3/IdentityIdentity.batch_normalization_3/moments/Squeeze:output:0^batch_normalization_3/truediv ^batch_normalization_3/truediv_1*
T0*
_output_shapes	
:��
 batch_normalization_3/Identity_1Identitybatch_normalization_3/Sqrt:y:0^batch_normalization_3/truediv ^batch_normalization_3/truediv_1*
T0*
_output_shapes	
:�{
 batch_normalization_3/Identity_2Identity'batch_normalization_3/Identity:output:0*
T0*
_output_shapes	
:�p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp1batch_normalization_3_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0)batch_normalization_3/Identity_2:output:0*
T0*
_output_shapes	
:��
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp1batch_normalization_3_sub_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp)^batch_normalization_3/sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0}
 batch_normalization_3/Identity_3Identity)batch_normalization_3/Identity_1:output:0*
T0*
_output_shapes	
:�r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp5batch_normalization_3_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:0)batch_normalization_3/Identity_3:output:0*
T0*
_output_shapes	
:��
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp5batch_normalization_3_maximum_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp-^batch_normalization_3/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
 batch_normalization_3/Identity_4Identity'batch_normalization_3/Identity:output:0&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1*
T0*
_output_shapes	
:��
 batch_normalization_3/Identity_5Identity0batch_normalization_3/moments/Squeeze_1:output:0&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1*
T0*
_output_shapes	
:�w
batch_normalization_3/renorm_rStopGradient!batch_normalization_3/truediv:z:0*
T0*
_output_shapes	
:�y
batch_normalization_3/renorm_dStopGradient#batch_normalization_3/truediv_1:z:0*
T0*
_output_shapes	
:��
(batch_normalization_3/mul/ReadVariableOpReadVariableOp1batch_normalization_3_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_3/mulMul'batch_normalization_3/renorm_r:output:00batch_normalization_3/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_3/mul_1/ReadVariableOpReadVariableOp1batch_normalization_3_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_3/mul_1Mul'batch_normalization_3/renorm_d:output:02batch_normalization_3/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_3/add_1/ReadVariableOpReadVariableOp3batch_normalization_3_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_3/add_1AddV2batch_normalization_3/mul_1:z:02batch_normalization_3/add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�r
-batch_normalization_3/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_3/AssignMovingAvg_2/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/AssignMovingAvg_2/subSub>batch_normalization_3/AssignMovingAvg_2/ReadVariableOp:value:0)batch_normalization_3/Identity_4:output:0*
T0*
_output_shapes	
:��
+batch_normalization_3/AssignMovingAvg_2/mulMul/batch_normalization_3/AssignMovingAvg_2/sub:z:06batch_normalization_3/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_3/AssignMovingAvg_2AssignSubVariableOp?batch_normalization_3_assignmovingavg_2_readvariableop_resource/batch_normalization_3/AssignMovingAvg_2/mul:z:07^batch_normalization_3/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0b
batch_normalization_3/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_3/add_2AddV2)batch_normalization_3/Identity_5:output:0&batch_normalization_3/add_2/y:output:0*
T0*
_output_shapes	
:�k
batch_normalization_3/Sqrt_2Sqrtbatch_normalization_3/add_2:z:0*
T0*
_output_shapes	
:�r
-batch_normalization_3/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_3/AssignMovingAvg_3/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_3/AssignMovingAvg_3/subSub>batch_normalization_3/AssignMovingAvg_3/ReadVariableOp:value:0 batch_normalization_3/Sqrt_2:y:0*
T0*
_output_shapes	
:��
+batch_normalization_3/AssignMovingAvg_3/mulMul/batch_normalization_3/AssignMovingAvg_3/sub:z:06batch_normalization_3/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_3/AssignMovingAvg_3AssignSubVariableOp?batch_normalization_3_assignmovingavg_3_readvariableop_resource/batch_normalization_3/AssignMovingAvg_3/mul:z:07^batch_normalization_3/AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
$batch_normalization_3/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_3_readvariableop_resource(^batch_normalization_3/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
*batch_normalization_3/mul_2/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_3_readvariableop_resource(^batch_normalization_3/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
batch_normalization_3/mul_2Mul,batch_normalization_3/ReadVariableOp:value:02batch_normalization_3/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
batch_normalization_3/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_3/sub_1Subbatch_normalization_3/mul_2:z:0&batch_normalization_3/sub_1/y:output:0*
T0*
_output_shapes	
:�i
batch_normalization_3/ReluRelubatch_normalization_3/sub_1:z:0*
T0*
_output_shapes	
:��
$batch_normalization_3/AssignNewValueAssignVariableOp-batch_normalization_3_assignnewvalue_resource(batch_normalization_3/Relu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0batch_normalization_3/mul:z:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/mul_1Muldense_2/Tanh:y:0'batch_normalization_3/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_3/batchnorm/subSubbatch_normalization_3/add_1:z:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_3/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_4/moments/meanMeandense_3/Tanh:y:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_3/Tanh:y:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 `
batch_normalization_4/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_4/addAddV20batch_normalization_4/moments/Squeeze_1:output:0$batch_normalization_4/add/y:output:0*
T0*
_output_shapes	
:�g
batch_normalization_4/SqrtSqrtbatch_normalization_4/add:z:0*
T0*
_output_shapes	
:�c
batch_normalization_4/Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:n
batch_normalization_4/Sqrt_1Sqrt'batch_normalization_4/Sqrt_1/x:output:0*
T0*
_output_shapes
: �
,batch_normalization_4/Maximum/ReadVariableOpReadVariableOp5batch_normalization_4_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_4/MaximumMaximum4batch_normalization_4/Maximum/ReadVariableOp:value:0 batch_normalization_4/Sqrt_1:y:0*
T0*
_output_shapes	
:��
batch_normalization_4/truedivRealDivbatch_normalization_4/Sqrt:y:0!batch_normalization_4/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
(batch_normalization_4/sub/ReadVariableOpReadVariableOp1batch_normalization_4_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_4/subSub.batch_normalization_4/moments/Squeeze:output:00batch_normalization_4/sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
batch_normalization_4/truediv_1RealDivbatch_normalization_4/sub:z:0!batch_normalization_4/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
batch_normalization_4/IdentityIdentity.batch_normalization_4/moments/Squeeze:output:0^batch_normalization_4/truediv ^batch_normalization_4/truediv_1*
T0*
_output_shapes	
:��
 batch_normalization_4/Identity_1Identitybatch_normalization_4/Sqrt:y:0^batch_normalization_4/truediv ^batch_normalization_4/truediv_1*
T0*
_output_shapes	
:�{
 batch_normalization_4/Identity_2Identity'batch_normalization_4/Identity:output:0*
T0*
_output_shapes	
:�p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp1batch_normalization_4_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0)batch_normalization_4/Identity_2:output:0*
T0*
_output_shapes	
:��
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp1batch_normalization_4_sub_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp)^batch_normalization_4/sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0}
 batch_normalization_4/Identity_3Identity)batch_normalization_4/Identity_1:output:0*
T0*
_output_shapes	
:�r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp5batch_normalization_4_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0)batch_normalization_4/Identity_3:output:0*
T0*
_output_shapes	
:��
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp5batch_normalization_4_maximum_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp-^batch_normalization_4/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
 batch_normalization_4/Identity_4Identity'batch_normalization_4/Identity:output:0&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes	
:��
 batch_normalization_4/Identity_5Identity0batch_normalization_4/moments/Squeeze_1:output:0&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes	
:�w
batch_normalization_4/renorm_rStopGradient!batch_normalization_4/truediv:z:0*
T0*
_output_shapes	
:�y
batch_normalization_4/renorm_dStopGradient#batch_normalization_4/truediv_1:z:0*
T0*
_output_shapes	
:��
(batch_normalization_4/mul/ReadVariableOpReadVariableOp1batch_normalization_4_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_4/mulMul'batch_normalization_4/renorm_r:output:00batch_normalization_4/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_4/mul_1/ReadVariableOpReadVariableOp1batch_normalization_4_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_4/mul_1Mul'batch_normalization_4/renorm_d:output:02batch_normalization_4/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_4/add_1/ReadVariableOpReadVariableOp3batch_normalization_4_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_4/add_1AddV2batch_normalization_4/mul_1:z:02batch_normalization_4/add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�r
-batch_normalization_4/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_2/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/AssignMovingAvg_2/subSub>batch_normalization_4/AssignMovingAvg_2/ReadVariableOp:value:0)batch_normalization_4/Identity_4:output:0*
T0*
_output_shapes	
:��
+batch_normalization_4/AssignMovingAvg_2/mulMul/batch_normalization_4/AssignMovingAvg_2/sub:z:06batch_normalization_4/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_4/AssignMovingAvg_2AssignSubVariableOp?batch_normalization_4_assignmovingavg_2_readvariableop_resource/batch_normalization_4/AssignMovingAvg_2/mul:z:07^batch_normalization_4/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0b
batch_normalization_4/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_4/add_2AddV2)batch_normalization_4/Identity_5:output:0&batch_normalization_4/add_2/y:output:0*
T0*
_output_shapes	
:�k
batch_normalization_4/Sqrt_2Sqrtbatch_normalization_4/add_2:z:0*
T0*
_output_shapes	
:�r
-batch_normalization_4/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_3/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_4/AssignMovingAvg_3/subSub>batch_normalization_4/AssignMovingAvg_3/ReadVariableOp:value:0 batch_normalization_4/Sqrt_2:y:0*
T0*
_output_shapes	
:��
+batch_normalization_4/AssignMovingAvg_3/mulMul/batch_normalization_4/AssignMovingAvg_3/sub:z:06batch_normalization_4/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_4/AssignMovingAvg_3AssignSubVariableOp?batch_normalization_4_assignmovingavg_3_readvariableop_resource/batch_normalization_4/AssignMovingAvg_3/mul:z:07^batch_normalization_4/AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
$batch_normalization_4/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_3_readvariableop_resource(^batch_normalization_4/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
*batch_normalization_4/mul_2/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_3_readvariableop_resource(^batch_normalization_4/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
batch_normalization_4/mul_2Mul,batch_normalization_4/ReadVariableOp:value:02batch_normalization_4/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
batch_normalization_4/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_4/sub_1Subbatch_normalization_4/mul_2:z:0&batch_normalization_4/sub_1/y:output:0*
T0*
_output_shapes	
:�i
batch_normalization_4/ReluRelubatch_normalization_4/sub_1:z:0*
T0*
_output_shapes	
:��
$batch_normalization_4/AssignNewValueAssignVariableOp-batch_normalization_4_assignnewvalue_resource(batch_normalization_4/Relu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0batch_normalization_4/mul:z:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/mul_1Muldense_3/Tanh:y:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_4/batchnorm/subSubbatch_normalization_4/add_1:z:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_4/MatMulMatMul)batch_normalization_4/batchnorm/add_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_5/moments/meanMeandense_4/Tanh:y:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_4/Tanh:y:03batch_normalization_5/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 `
batch_normalization_5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_5/addAddV20batch_normalization_5/moments/Squeeze_1:output:0$batch_normalization_5/add/y:output:0*
T0*
_output_shapes	
:�g
batch_normalization_5/SqrtSqrtbatch_normalization_5/add:z:0*
T0*
_output_shapes	
:�c
batch_normalization_5/Sqrt_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:n
batch_normalization_5/Sqrt_1Sqrt'batch_normalization_5/Sqrt_1/x:output:0*
T0*
_output_shapes
: �
,batch_normalization_5/Maximum/ReadVariableOpReadVariableOp5batch_normalization_5_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_5/MaximumMaximum4batch_normalization_5/Maximum/ReadVariableOp:value:0 batch_normalization_5/Sqrt_1:y:0*
T0*
_output_shapes	
:��
batch_normalization_5/truedivRealDivbatch_normalization_5/Sqrt:y:0!batch_normalization_5/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
(batch_normalization_5/sub/ReadVariableOpReadVariableOp1batch_normalization_5_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_5/subSub.batch_normalization_5/moments/Squeeze:output:00batch_normalization_5/sub/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
batch_normalization_5/truediv_1RealDivbatch_normalization_5/sub:z:0!batch_normalization_5/Maximum:z:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes	
:��
batch_normalization_5/IdentityIdentity.batch_normalization_5/moments/Squeeze:output:0^batch_normalization_5/truediv ^batch_normalization_5/truediv_1*
T0*
_output_shapes	
:��
 batch_normalization_5/Identity_1Identitybatch_normalization_5/Sqrt:y:0^batch_normalization_5/truediv ^batch_normalization_5/truediv_1*
T0*
_output_shapes	
:�{
 batch_normalization_5/Identity_2Identity'batch_normalization_5/Identity:output:0*
T0*
_output_shapes	
:�p
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp1batch_normalization_5_sub_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0)batch_normalization_5/Identity_2:output:0*
T0*
_output_shapes	
:��
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp1batch_normalization_5_sub_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp)^batch_normalization_5/sub/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0}
 batch_normalization_5/Identity_3Identity)batch_normalization_5/Identity_1:output:0*
T0*
_output_shapes	
:�r
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp5batch_normalization_5_maximum_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:0)batch_normalization_5/Identity_3:output:0*
T0*
_output_shapes	
:��
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp5batch_normalization_5_maximum_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp-^batch_normalization_5/Maximum/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
 batch_normalization_5/Identity_4Identity'batch_normalization_5/Identity:output:0&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1*
T0*
_output_shapes	
:��
 batch_normalization_5/Identity_5Identity0batch_normalization_5/moments/Squeeze_1:output:0&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1*
T0*
_output_shapes	
:�w
batch_normalization_5/renorm_rStopGradient!batch_normalization_5/truediv:z:0*
T0*
_output_shapes	
:�y
batch_normalization_5/renorm_dStopGradient#batch_normalization_5/truediv_1:z:0*
T0*
_output_shapes	
:��
(batch_normalization_5/mul/ReadVariableOpReadVariableOp1batch_normalization_5_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_5/mulMul'batch_normalization_5/renorm_r:output:00batch_normalization_5/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_5/mul_1/ReadVariableOpReadVariableOp1batch_normalization_5_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_5/mul_1Mul'batch_normalization_5/renorm_d:output:02batch_normalization_5/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*batch_normalization_5/add_1/ReadVariableOpReadVariableOp3batch_normalization_5_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batch_normalization_5/add_1AddV2batch_normalization_5/mul_1:z:02batch_normalization_5/add_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�r
-batch_normalization_5/AssignMovingAvg_2/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_5/AssignMovingAvg_2/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/AssignMovingAvg_2/subSub>batch_normalization_5/AssignMovingAvg_2/ReadVariableOp:value:0)batch_normalization_5/Identity_4:output:0*
T0*
_output_shapes	
:��
+batch_normalization_5/AssignMovingAvg_2/mulMul/batch_normalization_5/AssignMovingAvg_2/sub:z:06batch_normalization_5/AssignMovingAvg_2/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_5/AssignMovingAvg_2AssignSubVariableOp?batch_normalization_5_assignmovingavg_2_readvariableop_resource/batch_normalization_5/AssignMovingAvg_2/mul:z:07^batch_normalization_5/AssignMovingAvg_2/ReadVariableOp*
_output_shapes
 *
dtype0b
batch_normalization_5/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_5/add_2AddV2)batch_normalization_5/Identity_5:output:0&batch_normalization_5/add_2/y:output:0*
T0*
_output_shapes	
:�k
batch_normalization_5/Sqrt_2Sqrtbatch_normalization_5/add_2:z:0*
T0*
_output_shapes	
:�r
-batch_normalization_5/AssignMovingAvg_3/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_5/AssignMovingAvg_3/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_5/AssignMovingAvg_3/subSub>batch_normalization_5/AssignMovingAvg_3/ReadVariableOp:value:0 batch_normalization_5/Sqrt_2:y:0*
T0*
_output_shapes	
:��
+batch_normalization_5/AssignMovingAvg_3/mulMul/batch_normalization_5/AssignMovingAvg_3/sub:z:06batch_normalization_5/AssignMovingAvg_3/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_5/AssignMovingAvg_3AssignSubVariableOp?batch_normalization_5_assignmovingavg_3_readvariableop_resource/batch_normalization_5/AssignMovingAvg_3/mul:z:07^batch_normalization_5/AssignMovingAvg_3/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
$batch_normalization_5/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_3_readvariableop_resource(^batch_normalization_5/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
*batch_normalization_5/mul_2/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_3_readvariableop_resource(^batch_normalization_5/AssignMovingAvg_3*
_output_shapes	
:�*
dtype0�
batch_normalization_5/mul_2Mul,batch_normalization_5/ReadVariableOp:value:02batch_normalization_5/mul_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
batch_normalization_5/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
batch_normalization_5/sub_1Subbatch_normalization_5/mul_2:z:0&batch_normalization_5/sub_1/y:output:0*
T0*
_output_shapes	
:�i
batch_normalization_5/ReluRelubatch_normalization_5/sub_1:z:0*
T0*
_output_shapes	
:��
$batch_normalization_5/AssignNewValueAssignVariableOp-batch_normalization_5_assignnewvalue_resource(batch_normalization_5/Relu:activations:0*
_output_shapes
 *
dtype0*
validate_shape(j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:��
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0batch_normalization_5/mul:z:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/mul_1Muldense_4/Tanh:y:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
#batch_normalization_5/batchnorm/subSubbatch_normalization_5/add_1:z:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
output1/MatMul/ReadVariableOpReadVariableOp&output1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output1/MatMulMatMul)batch_normalization_5/batchnorm/add_1:z:0%output1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output1/BiasAdd/ReadVariableOpReadVariableOp'output1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output1/BiasAddBiasAddoutput1/MatMul:product:0&output1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
output1/SoftmaxSoftmaxoutput1/BiasAdd:output:0*
T0*'
_output_shapes
:���������^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater/GreaterGreateroutput1/Softmax:softmax:0"tf.math.greater/Greater/y:output:0*
T0*'
_output_shapes
:���������f
output2/ShapeShapetf.math.greater/Greater:z:0*
T0
*
_output_shapes
::��e
output2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
output2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
output2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
output2/strided_sliceStridedSliceoutput2/Shape:output:0$output2/strided_slice/stack:output:0&output2/strided_slice/stack_1:output:0&output2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
output2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
output2/Reshape/shapePackoutput2/strided_slice:output:0 output2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
output2/ReshapeReshapetf.math.greater/Greater:z:0output2/Reshape/shape:output:0*
T0
*'
_output_shapes
:���������h
IdentityIdentityoutput1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������i

Identity_1Identityoutput2/Reshape:output:0^NoOp*
T0
*'
_output_shapes
:����������%
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp&^batch_normalization/AssignMovingAvg_25^batch_normalization/AssignMovingAvg_2/ReadVariableOp&^batch_normalization/AssignMovingAvg_35^batch_normalization/AssignMovingAvg_3/ReadVariableOp#^batch_normalization/AssignNewValue+^batch_normalization/Maximum/ReadVariableOp#^batch_normalization/ReadVariableOp)^batch_normalization/add_1/ReadVariableOp'^batch_normalization/mul/ReadVariableOp)^batch_normalization/mul_1/ReadVariableOp)^batch_normalization/mul_2/ReadVariableOp'^batch_normalization/sub/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_27^batch_normalization_1/AssignMovingAvg_2/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_37^batch_normalization_1/AssignMovingAvg_3/ReadVariableOp%^batch_normalization_1/AssignNewValue-^batch_normalization_1/Maximum/ReadVariableOp%^batch_normalization_1/ReadVariableOp+^batch_normalization_1/add_1/ReadVariableOp)^batch_normalization_1/mul/ReadVariableOp+^batch_normalization_1/mul_1/ReadVariableOp+^batch_normalization_1/mul_2/ReadVariableOp)^batch_normalization_1/sub/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_27^batch_normalization_2/AssignMovingAvg_2/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_37^batch_normalization_2/AssignMovingAvg_3/ReadVariableOp%^batch_normalization_2/AssignNewValue-^batch_normalization_2/Maximum/ReadVariableOp%^batch_normalization_2/ReadVariableOp+^batch_normalization_2/add_1/ReadVariableOp)^batch_normalization_2/mul/ReadVariableOp+^batch_normalization_2/mul_1/ReadVariableOp+^batch_normalization_2/mul_2/ReadVariableOp)^batch_normalization_2/sub/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_27^batch_normalization_3/AssignMovingAvg_2/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_37^batch_normalization_3/AssignMovingAvg_3/ReadVariableOp%^batch_normalization_3/AssignNewValue-^batch_normalization_3/Maximum/ReadVariableOp%^batch_normalization_3/ReadVariableOp+^batch_normalization_3/add_1/ReadVariableOp)^batch_normalization_3/mul/ReadVariableOp+^batch_normalization_3/mul_1/ReadVariableOp+^batch_normalization_3/mul_2/ReadVariableOp)^batch_normalization_3/sub/ReadVariableOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_27^batch_normalization_4/AssignMovingAvg_2/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_37^batch_normalization_4/AssignMovingAvg_3/ReadVariableOp%^batch_normalization_4/AssignNewValue-^batch_normalization_4/Maximum/ReadVariableOp%^batch_normalization_4/ReadVariableOp+^batch_normalization_4/add_1/ReadVariableOp)^batch_normalization_4/mul/ReadVariableOp+^batch_normalization_4/mul_1/ReadVariableOp+^batch_normalization_4/mul_2/ReadVariableOp)^batch_normalization_4/sub/ReadVariableOp&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_27^batch_normalization_5/AssignMovingAvg_2/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_37^batch_normalization_5/AssignMovingAvg_3/ReadVariableOp%^batch_normalization_5/AssignNewValue-^batch_normalization_5/Maximum/ReadVariableOp%^batch_normalization_5/ReadVariableOp+^batch_normalization_5/add_1/ReadVariableOp)^batch_normalization_5/mul/ReadVariableOp+^batch_normalization_5/mul_1/ReadVariableOp+^batch_normalization_5/mul_2/ReadVariableOp)^batch_normalization_5/sub/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^output1/BiasAdd/ReadVariableOp^output1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_2%batch_normalization/AssignMovingAvg_22l
4batch_normalization/AssignMovingAvg_2/ReadVariableOp4batch_normalization/AssignMovingAvg_2/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_3%batch_normalization/AssignMovingAvg_32l
4batch_normalization/AssignMovingAvg_3/ReadVariableOp4batch_normalization/AssignMovingAvg_3/ReadVariableOp2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2X
*batch_normalization/Maximum/ReadVariableOp*batch_normalization/Maximum/ReadVariableOp2H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2T
(batch_normalization/add_1/ReadVariableOp(batch_normalization/add_1/ReadVariableOp2P
&batch_normalization/mul/ReadVariableOp&batch_normalization/mul/ReadVariableOp2T
(batch_normalization/mul_1/ReadVariableOp(batch_normalization/mul_1/ReadVariableOp2T
(batch_normalization/mul_2/ReadVariableOp(batch_normalization/mul_2/ReadVariableOp2P
&batch_normalization/sub/ReadVariableOp&batch_normalization/sub/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_2'batch_normalization_1/AssignMovingAvg_22p
6batch_normalization_1/AssignMovingAvg_2/ReadVariableOp6batch_normalization_1/AssignMovingAvg_2/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_3'batch_normalization_1/AssignMovingAvg_32p
6batch_normalization_1/AssignMovingAvg_3/ReadVariableOp6batch_normalization_1/AssignMovingAvg_3/ReadVariableOp2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2\
,batch_normalization_1/Maximum/ReadVariableOp,batch_normalization_1/Maximum/ReadVariableOp2L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2X
*batch_normalization_1/add_1/ReadVariableOp*batch_normalization_1/add_1/ReadVariableOp2T
(batch_normalization_1/mul/ReadVariableOp(batch_normalization_1/mul/ReadVariableOp2X
*batch_normalization_1/mul_1/ReadVariableOp*batch_normalization_1/mul_1/ReadVariableOp2X
*batch_normalization_1/mul_2/ReadVariableOp*batch_normalization_1/mul_2/ReadVariableOp2T
(batch_normalization_1/sub/ReadVariableOp(batch_normalization_1/sub/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_2'batch_normalization_2/AssignMovingAvg_22p
6batch_normalization_2/AssignMovingAvg_2/ReadVariableOp6batch_normalization_2/AssignMovingAvg_2/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_3'batch_normalization_2/AssignMovingAvg_32p
6batch_normalization_2/AssignMovingAvg_3/ReadVariableOp6batch_normalization_2/AssignMovingAvg_3/ReadVariableOp2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2\
,batch_normalization_2/Maximum/ReadVariableOp,batch_normalization_2/Maximum/ReadVariableOp2L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2X
*batch_normalization_2/add_1/ReadVariableOp*batch_normalization_2/add_1/ReadVariableOp2T
(batch_normalization_2/mul/ReadVariableOp(batch_normalization_2/mul/ReadVariableOp2X
*batch_normalization_2/mul_1/ReadVariableOp*batch_normalization_2/mul_1/ReadVariableOp2X
*batch_normalization_2/mul_2/ReadVariableOp*batch_normalization_2/mul_2/ReadVariableOp2T
(batch_normalization_2/sub/ReadVariableOp(batch_normalization_2/sub/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_2'batch_normalization_3/AssignMovingAvg_22p
6batch_normalization_3/AssignMovingAvg_2/ReadVariableOp6batch_normalization_3/AssignMovingAvg_2/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_3'batch_normalization_3/AssignMovingAvg_32p
6batch_normalization_3/AssignMovingAvg_3/ReadVariableOp6batch_normalization_3/AssignMovingAvg_3/ReadVariableOp2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2\
,batch_normalization_3/Maximum/ReadVariableOp,batch_normalization_3/Maximum/ReadVariableOp2L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2X
*batch_normalization_3/add_1/ReadVariableOp*batch_normalization_3/add_1/ReadVariableOp2T
(batch_normalization_3/mul/ReadVariableOp(batch_normalization_3/mul/ReadVariableOp2X
*batch_normalization_3/mul_1/ReadVariableOp*batch_normalization_3/mul_1/ReadVariableOp2X
*batch_normalization_3/mul_2/ReadVariableOp*batch_normalization_3/mul_2/ReadVariableOp2T
(batch_normalization_3/sub/ReadVariableOp(batch_normalization_3/sub/ReadVariableOp2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_2'batch_normalization_4/AssignMovingAvg_22p
6batch_normalization_4/AssignMovingAvg_2/ReadVariableOp6batch_normalization_4/AssignMovingAvg_2/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_3'batch_normalization_4/AssignMovingAvg_32p
6batch_normalization_4/AssignMovingAvg_3/ReadVariableOp6batch_normalization_4/AssignMovingAvg_3/ReadVariableOp2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2\
,batch_normalization_4/Maximum/ReadVariableOp,batch_normalization_4/Maximum/ReadVariableOp2L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2X
*batch_normalization_4/add_1/ReadVariableOp*batch_normalization_4/add_1/ReadVariableOp2T
(batch_normalization_4/mul/ReadVariableOp(batch_normalization_4/mul/ReadVariableOp2X
*batch_normalization_4/mul_1/ReadVariableOp*batch_normalization_4/mul_1/ReadVariableOp2X
*batch_normalization_4/mul_2/ReadVariableOp*batch_normalization_4/mul_2/ReadVariableOp2T
(batch_normalization_4/sub/ReadVariableOp(batch_normalization_4/sub/ReadVariableOp2N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_2'batch_normalization_5/AssignMovingAvg_22p
6batch_normalization_5/AssignMovingAvg_2/ReadVariableOp6batch_normalization_5/AssignMovingAvg_2/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_3'batch_normalization_5/AssignMovingAvg_32p
6batch_normalization_5/AssignMovingAvg_3/ReadVariableOp6batch_normalization_5/AssignMovingAvg_3/ReadVariableOp2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2\
,batch_normalization_5/Maximum/ReadVariableOp,batch_normalization_5/Maximum/ReadVariableOp2L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2X
*batch_normalization_5/add_1/ReadVariableOp*batch_normalization_5/add_1/ReadVariableOp2T
(batch_normalization_5/mul/ReadVariableOp(batch_normalization_5/mul/ReadVariableOp2X
*batch_normalization_5/mul_1/ReadVariableOp*batch_normalization_5/mul_1/ReadVariableOp2X
*batch_normalization_5/mul_2/ReadVariableOp*batch_normalization_5/mul_2/ReadVariableOp2T
(batch_normalization_5/sub/ReadVariableOp(batch_normalization_5/sub/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
output1/BiasAdd/ReadVariableOpoutput1/BiasAdd/ReadVariableOp2>
output1/MatMul/ReadVariableOpoutput1/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
input1/
serving_default_input1:0���������
9
input2/
serving_default_input2:0���������;
output10
StatefulPartitionedCall:0���������;
output20
StatefulPartitionedCall:1
���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer-15
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&renorm_clipping
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,moving_stddev
-renorm_mean
.renorm_stddev"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=renorm_clipping
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Cmoving_stddev
Drenorm_mean
Erenorm_stddev"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
Trenorm_clipping
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Zmoving_stddev
[renorm_mean
\renorm_stddev"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
krenorm_clipping
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qmoving_stddev
rrenorm_mean
srenorm_stddev"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�renorm_clipping
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�moving_stddev
�renorm_mean
�renorm_stddev"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�renorm_clipping
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�moving_stddev
�renorm_mean
�renorm_stddev"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(0
)1
*2
+3
,4
-5
.6
57
68
?9
@10
A11
B12
C13
D14
E15
L16
M17
V18
W19
X20
Y21
Z22
[23
\24
c25
d26
m27
n28
o29
p30
q31
r32
s33
z34
{35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53"
trackable_list_wrapper
�
(0
)1
52
63
?4
@5
L6
M7
V8
W9
c10
d11
m12
n13
z14
{15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
$__inference_model_layer_call_fn_1854
$__inference_model_layer_call_fn_2758
$__inference_model_layer_call_fn_2874
$__inference_model_layer_call_fn_2370�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
?__inference_model_layer_call_and_return_conditional_losses_3031
?__inference_model_layer_call_and_return_conditional_losses_3530
?__inference_model_layer_call_and_return_conditional_losses_2465
?__inference_model_layer_call_and_return_conditional_losses_2596�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
__inference__wrapped_model_798input1input2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_concatenate_layer_call_fn_3536�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_concatenate_layer_call_and_return_conditional_losses_3543�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
Q
(0
)1
*2
+3
,4
-5
.6"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_batch_normalization_layer_call_fn_3556
2__inference_batch_normalization_layer_call_fn_3575�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3595
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3672�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
1:/ (2!batch_normalization/moving_stddev
/:- (2batch_normalization/renorm_mean
1:/ (2!batch_normalization/renorm_stddev
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_dense_layer_call_fn_3681�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_dense_layer_call_and_return_conditional_losses_3692�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2dense/kernel
:�2
dense/bias
Q
?0
@1
A2
B3
C4
D5
E6"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_1_layer_call_fn_3705
4__inference_batch_normalization_1_layer_call_fn_3724�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3744
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3821�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
*:(�2batch_normalization_1/gamma
):'�2batch_normalization_1/beta
2:0� (2!batch_normalization_1/moving_mean
6:4� (2%batch_normalization_1/moving_variance
4:2� (2#batch_normalization_1/moving_stddev
2:0� (2!batch_normalization_1/renorm_mean
4:2� (2#batch_normalization_1/renorm_stddev
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_1_layer_call_fn_3830�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_1_layer_call_and_return_conditional_losses_3841�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_1/kernel
:�2dense_1/bias
Q
V0
W1
X2
Y3
Z4
[5
\6"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_2_layer_call_fn_3854
4__inference_batch_normalization_2_layer_call_fn_3873�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3893
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3970�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
4:2� (2#batch_normalization_2/moving_stddev
2:0� (2!batch_normalization_2/renorm_mean
4:2� (2#batch_normalization_2/renorm_stddev
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_2_layer_call_fn_3979�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_2_layer_call_and_return_conditional_losses_3990�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_2/kernel
:�2dense_2/bias
Q
m0
n1
o2
p3
q4
r5
s6"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_3_layer_call_fn_4003
4__inference_batch_normalization_3_layer_call_fn_4022�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4042
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4119�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
*:(�2batch_normalization_3/gamma
):'�2batch_normalization_3/beta
2:0� (2!batch_normalization_3/moving_mean
6:4� (2%batch_normalization_3/moving_variance
4:2� (2#batch_normalization_3/moving_stddev
2:0� (2!batch_normalization_3/renorm_mean
4:2� (2#batch_normalization_3/renorm_stddev
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_3_layer_call_fn_4128�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_3_layer_call_and_return_conditional_losses_4139�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_3/kernel
:�2dense_3/bias
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_4_layer_call_fn_4152
4__inference_batch_normalization_4_layer_call_fn_4171�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4191
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4268�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
*:(�2batch_normalization_4/gamma
):'�2batch_normalization_4/beta
2:0� (2!batch_normalization_4/moving_mean
6:4� (2%batch_normalization_4/moving_variance
4:2� (2#batch_normalization_4/moving_stddev
2:0� (2!batch_normalization_4/renorm_mean
4:2� (2#batch_normalization_4/renorm_stddev
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_4_layer_call_fn_4277�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_4_layer_call_and_return_conditional_losses_4288�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_4/kernel
:�2dense_4/bias
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_5_layer_call_fn_4301
4__inference_batch_normalization_5_layer_call_fn_4320�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4340
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4417�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
*:(�2batch_normalization_5/gamma
):'�2batch_normalization_5/beta
2:0� (2!batch_normalization_5/moving_mean
6:4� (2%batch_normalization_5/moving_variance
4:2� (2#batch_normalization_5/moving_stddev
2:0� (2!batch_normalization_5/renorm_mean
4:2� (2#batch_normalization_5/renorm_stddev
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_output1_layer_call_fn_4426�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_output1_layer_call_and_return_conditional_losses_4437�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:	�2output1/kernel
:2output1/bias
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_output2_layer_call_fn_4442�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_output2_layer_call_and_return_conditional_losses_4454�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
*0
+1
,2
-3
.4
A5
B6
C7
D8
E9
X10
Y11
Z12
[13
\14
o15
p16
q17
r18
s19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_1854input1input2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_2758inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_2874inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_2370input1input2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_3031inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_3530inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_2465input1input2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_2596input1input2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_2678input1input2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_concatenate_layer_call_fn_3536inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_concatenate_layer_call_and_return_conditional_losses_3543inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
C
*0
+1
,2
-3
.4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_batch_normalization_layer_call_fn_3556inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_batch_normalization_layer_call_fn_3575inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3595inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3672inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_dense_layer_call_fn_3681inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dense_layer_call_and_return_conditional_losses_3692inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
C
A0
B1
C2
D3
E4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_1_layer_call_fn_3705inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_1_layer_call_fn_3724inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3744inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3821inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_1_layer_call_fn_3830inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_1_layer_call_and_return_conditional_losses_3841inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
C
X0
Y1
Z2
[3
\4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_2_layer_call_fn_3854inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_2_layer_call_fn_3873inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3893inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3970inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_2_layer_call_fn_3979inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_2_layer_call_and_return_conditional_losses_3990inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
C
o0
p1
q2
r3
s4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_3_layer_call_fn_4003inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_3_layer_call_fn_4022inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4042inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4119inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_3_layer_call_fn_4128inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_3_layer_call_and_return_conditional_losses_4139inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_4_layer_call_fn_4152inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_4_layer_call_fn_4171inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4191inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4268inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_4_layer_call_fn_4277inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_4_layer_call_and_return_conditional_losses_4288inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_5_layer_call_fn_4301inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_5_layer_call_fn_4320inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4340inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4417inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_output1_layer_call_fn_4426inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_output1_layer_call_and_return_conditional_losses_4437inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_output2_layer_call_fn_4442inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_output2_layer_call_and_return_conditional_losses_4454inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_798�0+(*)56B?A@LMYVXWcdpmonz{������������V�S
L�I
G�D
 �
input1���������
 �
input2���������
� "_�\
,
output1!�
output1���������
,
output2!�
output2���������
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3744kB?A@4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3821nED?@ACB4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
4__inference_batch_normalization_1_layer_call_fn_3705`B?A@4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
4__inference_batch_normalization_1_layer_call_fn_3724cED?@ACB4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3893kYVXW4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3970n\[VWXZY4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
4__inference_batch_normalization_2_layer_call_fn_3854`YVXW4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
4__inference_batch_normalization_2_layer_call_fn_3873c\[VWXZY4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4042kpmon4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4119nsrmnoqp4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
4__inference_batch_normalization_3_layer_call_fn_4003`pmon4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
4__inference_batch_normalization_3_layer_call_fn_4022csrmnoqp4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4191o����4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4268u�������4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
4__inference_batch_normalization_4_layer_call_fn_4152d����4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
4__inference_batch_normalization_4_layer_call_fn_4171j�������4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4340o����4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4417u�������4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
4__inference_batch_normalization_5_layer_call_fn_4301d����4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
4__inference_batch_normalization_5_layer_call_fn_4320j�������4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3595i+(*)3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3672l.-()*,+3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
2__inference_batch_normalization_layer_call_fn_3556^+(*)3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
2__inference_batch_normalization_layer_call_fn_3575a.-()*,+3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
E__inference_concatenate_layer_call_and_return_conditional_losses_3543�Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� ",�)
"�
tensor_0���������
� �
*__inference_concatenate_layer_call_fn_3536Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� "!�
unknown����������
A__inference_dense_1_layer_call_and_return_conditional_losses_3841eLM0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_1_layer_call_fn_3830ZLM0�-
&�#
!�
inputs����������
� ""�
unknown�����������
A__inference_dense_2_layer_call_and_return_conditional_losses_3990ecd0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_2_layer_call_fn_3979Zcd0�-
&�#
!�
inputs����������
� ""�
unknown�����������
A__inference_dense_3_layer_call_and_return_conditional_losses_4139ez{0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_3_layer_call_fn_4128Zz{0�-
&�#
!�
inputs����������
� ""�
unknown�����������
A__inference_dense_4_layer_call_and_return_conditional_losses_4288g��0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_4_layer_call_fn_4277\��0�-
&�#
!�
inputs����������
� ""�
unknown�����������
?__inference_dense_layer_call_and_return_conditional_losses_3692d56/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
$__inference_dense_layer_call_fn_3681Y56/�,
%�"
 �
inputs���������
� ""�
unknown�����������
?__inference_model_layer_call_and_return_conditional_losses_2465�0+(*)56B?A@LMYVXWcdpmonz{������������^�[
T�Q
G�D
 �
input1���������
 �
input2���������
p 

 
� "Y�V
O�L
$�!

tensor_0_0���������
$�!

tensor_0_1���������

� �
?__inference_model_layer_call_and_return_conditional_losses_2596�H.-()*,+56ED?@ACBLM\[VWXZYcdsrmnoqpz{������������������^�[
T�Q
G�D
 �
input1���������
 �
input2���������
p

 
� "Y�V
O�L
$�!

tensor_0_0���������
$�!

tensor_0_1���������

� �
?__inference_model_layer_call_and_return_conditional_losses_3031�0+(*)56B?A@LMYVXWcdpmonz{������������b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p 

 
� "Y�V
O�L
$�!

tensor_0_0���������
$�!

tensor_0_1���������

� �
?__inference_model_layer_call_and_return_conditional_losses_3530�H.-()*,+56ED?@ACBLM\[VWXZYcdsrmnoqpz{������������������b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p

 
� "Y�V
O�L
$�!

tensor_0_0���������
$�!

tensor_0_1���������

� �
$__inference_model_layer_call_fn_1854�0+(*)56B?A@LMYVXWcdpmonz{������������^�[
T�Q
G�D
 �
input1���������
 �
input2���������
p 

 
� "K�H
"�
tensor_0���������
"�
tensor_1���������
�
$__inference_model_layer_call_fn_2370�H.-()*,+56ED?@ACBLM\[VWXZYcdsrmnoqpz{������������������^�[
T�Q
G�D
 �
input1���������
 �
input2���������
p

 
� "K�H
"�
tensor_0���������
"�
tensor_1���������
�
$__inference_model_layer_call_fn_2758�0+(*)56B?A@LMYVXWcdpmonz{������������b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p 

 
� "K�H
"�
tensor_0���������
"�
tensor_1���������
�
$__inference_model_layer_call_fn_2874�H.-()*,+56ED?@ACBLM\[VWXZYcdsrmnoqpz{������������������b�_
X�U
K�H
"�
inputs_0���������
"�
inputs_1���������
p

 
� "K�H
"�
tensor_0���������
"�
tensor_1���������
�
A__inference_output1_layer_call_and_return_conditional_losses_4437f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
&__inference_output1_layer_call_fn_4426[��0�-
&�#
!�
inputs����������
� "!�
unknown����������
A__inference_output2_layer_call_and_return_conditional_losses_4454_/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������

� ~
&__inference_output2_layer_call_fn_4442T/�,
%�"
 �
inputs���������

� "!�
unknown���������
�
"__inference_signature_wrapper_2678�0+(*)56B?A@LMYVXWcdpmonz{������������e�b
� 
[�X
*
input1 �
input1���������
*
input2 �
input2���������"_�\
,
output1!�
output1���������
,
output2!�
output2���������
