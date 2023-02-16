print ("a in")
import sys
print ("b imported: %s" % ("b" in sys.modules, ))
import b
print ("a out")