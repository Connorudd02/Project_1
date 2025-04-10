# Exercise 0.4.2 - using the VScode editor (ignore if you are not )
#

## Option I
# Standard python code will look like the 4 code lines below
# and this is the way all scripts are 
# displayed/implemented in this course
#
# To run the whole file you need to press 
# the run symbol, F5 or Ctrl+F5 (debug)
#
a = 1
print(a)
b = a + 1
print(b)

#%%[markdown] 
# ## Option II
# If you prefer, you can get an 
# interactive experience in VScode if 
# you use the #%% tag, e.g.

#%%[markdown]  
# This is a cell you can run without running the rest of the code
# and get some pretty formatted output. 
c = 2
print(c)

#%%[markdown] 
# This is an other cell
# In the interactive mode you can even use markdown and include equations in teh output
# $$ b = c + \frac{1}{2} $$
#
# Note here the cell depends on the value of a variable from an other cell
# whcih must be run before this cell.
#
b = c + 1/2
print(b)


#%%[markdown]
# If you want this behavior you simply add the #%% or #%%[markdown] to the provided
# scripts in for you useful places.
#

