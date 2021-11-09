
import cgi,os

print('content-type:text/html\r\n\r\n')
form=cgi.FieldStorage()
event=str(form.getvalue("name"))
fle = form['image']
fn=os.path.basename(fle.image)
open("C:/xampp/htdocs/module1/picture/"+fn,"wb").write(fle.file.read())
print('<html>')
print('<body><center>')
print('<h1>Event\n(%s)</h1>'%event)
print('<img src=pictures/%s>'%fn)
print('</center><body></html>')
