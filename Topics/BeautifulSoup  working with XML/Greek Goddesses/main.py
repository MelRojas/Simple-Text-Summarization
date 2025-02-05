from bs4 import BeautifulSoup

goddesses = '<?xml version="1.0" encoding="UTF-8"?> \
<goddesses> \
  <goddess>Hestia</goddess> \
  <goddess>Hebe</goddess> \
  <goddess>Nemesis</goddess> \
  <goddess>Leto</goddess> \
  <goddess>Rhea</goddess> \
  <goddess>Aphrodite</goddess> \
  <goddess>Demeter</goddess> \
  <goddess>Artemis</goddess> \
  <goddess>Hera</goddess> \
  <goddess>Athena</goddess> \
  <goddess>Cybele</goddess> \
  <goddess>Persephone</goddess> \
  <goddess>Nyx</goddess> \
  <goddess>Selene</goddess> \
  <goddess>Tyche</goddess> \
  <goddess>Maia</goddess> \
  <goddess>Lachesis</goddess> \
  <goddess>Iris</goddess> \
  <goddess>Nike</goddess> \
  <goddess>Keres</goddess> \
</goddesses>'

# put your code here
soup = BeautifulSoup(goddesses, 'xml')

goddesses = soup.find_all('goddess')

for goddess in goddesses:
    print(goddess.text)