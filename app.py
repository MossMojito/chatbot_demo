import streamlit as st
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Configuration ---
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL = 'google/flan-t5-small'

# !!! EDIT THESE TWO VARIABLES !!!
FAISS_INDEX_PATH = "km1139171_index.faiss"  # <--- 1. Put your FAISS file name here
documents = ['Topic: โปรโมชั่นภายในประเทศ (อัตรา/เงื่อนไข), โปรโมชั่นภายในประเทศ '
 '(อัตรา/เงื่อนไข)',
 'Sub Topic: แพ็กเกจเสริมรายเดือน (Postpaid) สมัครรายครั้ง (One-Time), '
 'แพ็กเกจเสริมเติมเงิน (Prepaid) สมัครรายครั้ง (One-Time)',
 'More Info1: แพ็กเสริมเน็ต, แพ็กเสริมเน็ต',
 'More Info2: เน็ต 5G 1GB หลังจากนั้นเร็ว 6Mbps 55 บาท(รวม Vat) ใช้ได้ 1 วัน, '
 '3803085 - เน็ต 5G 1GB หลังจากนั้นเร็ว 6Mbps 55 บาท(รวม Vat) ใช้ได้ 1 วัน',
 'ชื่อ แพ็กเกจเสริม เน็ต5G เล่นได้ 1GB หลังจากนั้นใช้ต่อที่ความเร็ว 6 Mbps '
 'ใช้ได้ 1 วัน ค่าบริการ 55 บาท หักแบบรายครั้ง [Prepaid & Postpaid] Concept '
 'Package แพ็กเกจเสริมอินเทอร์เน็ต 55 บาท + ใช้งานเน็ตที่ความเร็ว 5G/4G '
 'เป็นจำนวน 1 GB หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps + ค่าบริการ '
 '51.40 บาท (ไม่รวม VAT)  ll  ราคารวม VAT 55 บาท + '
 'แพ็กเกจนี้หักค่าบริการแบบรายครั้ง + ลูกค้าเริ่มใช้งานได้ หลังจากได้รับ SMS '
 'ยืนยันการสมัครแพ็กเกจ + ใช้งานได้นาน 1 วัน กลุ่มลูกค้า + ลูกค้า AIS '
 'บุคคลธรรมดา / ลูกค้านิติบุคคล (ยกเว้นลูกค้า MVPN) ♦  ระบบรายเดือน ♦  '
 'ระบบเติมเงิน One-2-Call (CBS) + ลูกค้าที่ใช้สมาร์ทโฟนทั้ง 5G/4G '
 'ระยะเวลาสมัคร ตั้งแต่วันที่ 31 มกราคม 2567 เป็นต้นไป ช่องทางการสมัคร ช่องทาง '
 'SLA การสมัคร 1. AIS Call Center 2. USSD *777*7722# โทรออก (ยกเว้น : '
 'ลูกค้ารายเดือน ที่จดทะเบียนในนามนิติบุคคล) 3. AIS Shop รวม Serenade Club 4. '
 'Telewiz 5. Easy app (ROM) เริ่มวันที่ 3 เม.ย. 2567 เป็นต้นไป มีผลทันที '
 'ใช้งานได้หลังจากได้รับ SMS ยืนยัน หมายเหตุ : '
 '**ลูกค้านิติบุคคลสามารถรับสิทธิ์สมัครแพ็กเกจเสริมได้ตามช่องทางการสมัครของทางนิติบุคคล** '
 'ช่องทางการตรวจสอบ ยอดการใช้งาน + my AIS + บริการ USSD *121# + บริการ IVR '
 'Single Number *121 รายละเอียด แพ็กเกจเสริมเน็ต แบบรายครั้ง Postpaid Product '
 'Code P240122840 Promotion Name 5G_Internet In55B 1GB UL 1Day Prepaid Feature '
 'Code 3803085 Promotion Name 5G Internet 55B 1GB UL 1Day ** ห้ามพนักงาน '
 'Callcenter สมัครโดยใช้ Code ROM ข้อมูลเพื่อทราบ กรณีสมัครแพ็กเกจผ่าน ROM '
 'เท่านั้น Feature Code 3803169 Promotion Name 5G Internet 55B 1GB UL 1Day_ROM '
 'ประเภท ใช้งานอินเทอร์เน็ต 5G/4G ค่าบริการ (บาท) ใช้งาน (วัน) On-top Package '
 '(One Time Charge) ใช้งานเน็ตที่ความเร็ว 5G/4G เป็นจำนวน 1 GB '
 'หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps 55 บาท (รวม VAT) 1 วัน ( '
 'Time to Time ) 51.40 บาท (ไม่รวม VAT) เงื่อนไขการใช้งาน เงื่อนไขแพ็กเกจเสริม '
 '1. อัตราค่าบริการที่กำหนดเป็นอัตรารวมภาษีมูลค่าเพิ่ม 2. '
 'สิทธิ์การใช้งานแพ็กเกจเสริม สามารถใช้งานได้หลังจากได้รับ SMS ยืนยัน 3. '
 'สำหรับผู้ใช้สมาร์ทโฟนที่รองรับ 5G พร้อมทั้งอยู่ในพื้นที่ให้บริการ 5G '
 'สิทธิ์การใช้งานแพ็กเสริมเน็ต ใช้ได้ทั้ง 5G/4G '
 'โดยความเร็วสูงสุดของแพ็กเกจเป็น 5G/4G 4. สำหรับผู้ใช้สมาร์ทโฟนที่ '
 'ไม่รับรองรับ 5G สามารถสมัครแพ็กเกจเสริมนี้ได้ '
 'โดยสิทธิ์การใช้งานแพ็กเสริมเน็ต ใช้ได้ทั้ง 4G '
 'โดยความเร็วสูงสุดของแพ็กเกจเป็น 4G 5. '
 'อัตราค่าบริการอินเทอร์เน็ตคิดตามการใช้งานเป็นกิโลไบท์ (KB) '
 'ส่วนเกินคิดค่าบริการตามแพ็กเกจหลัก 6. ผู้ใช้บริการ MultiSIM '
 'ไม่สามารถสมัครแพ็กเกจอินเทอร์เน็ตไม่จำกัด '
 'หรือแพ็กเกจอินเทอร์เน็ตที่มีการใช้งานต่อเนื่องได้ 7. '
 'สิทธิ์การใช้งานและอัตราค่าบริการที่กำหนด สำหรับการใช้งานภายในประเทศ '
 'และการใช้งานปกติเท่านั้น 8. เพื่อเป็นการรักษามาตรฐานคุณภาพการให้บริการ '
 'และเพื่อให้ผู้ใช้บริการโดยรวมสามารถใช้งานได้ อย่างมีประสิทธิภาพ บริษัทฯ '
 'ขอสงวนสิทธิ์ในการบริหารจัดการเครือข่ายตามความเหมาะสม เช่น '
 'จำกัดหรือลดความเร็วในกรณีที่พบว่ามีการรับส่งข้อมูลในปริมาณมากอย่างต่อเนื่อง '
 'การใช้งาน BitTorrent การแชร์เน็ตผ่าน Hotspot การดาวน์โหลด และ/หรือ '
 'อัปโหลดไฟล์ขนาดใหญ่ หรือใช้งานในเชิงพาณิชย์ '
 'หรือมีการใช้ซึ่งส่งผลกระทบหรือก่อให้เกิดความไม่เป็นธรรมต่อผู้ใช้บริการรายอื่น '
 'หรือกระทบต่อเครือข่ายการให้บริการโดยรวมของบริษัท วิธีการเปิดใช้บริการ AIS 5G '
 'สำหรับผู้ใช้สมาร์ทโฟนที่รองรับ เพียงเปิดบริการ AIS 5G สามารถใช้ AIS 5G '
 'กับแพ็กเกจอินเทอร์เน็ตปัจจุบัน โดยไม่มีค่าใช้จ่ายเพิ่มเติม + '
 'กรณีลูกค้าซื้อสมาร์ทโฟน 5G จาก AIS จะได้รับบริการ AIS 5G อัตโนมัติ + '
 'ลูกค้าที่สมัครแพ็กเกจเสริมสำเร็จ จะได้รับการเปิดบริการ 5G ให้โดยอัตโนมัติ '
 'เงื่อนไขการเปิดบริการ AIS 5G + สงวนสิทธิ์การเปิดบริการ 1 หมายเลข/1 IMEI '
 'เครื่องเท่านั้น ** กรณีหมายเลขหรือเครื่องเคยเปิดบริการไปแล้ว '
 'ไม่สามารถรับสิทธิ์ซ้ำได้อีก ** SMS ยืนยัน การสมัครแพ็กสำเร็จ Postpaid '
 'แพ็กเกจเสริม 5G ค่าบริการ 55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด '
 'เร็ว 6Mbps ใช้ได้ 24ชม. เริ่ม[Start date] ถึง [End date] ซื้อแพ็กเน็ตเพิ่ม '
 'คลิก https://m.ais.co.th/5Gontop 5G On-top package 55Baht(Inc.VAT). You have '
 'got internet 1GB and unlimited internet at speed 6Mbps valid for 24hours. '
 'Start[Start date] to [End date]. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop Prepaid คุณเริ่มใช้แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'ซื้อแพ็กเน็ตเพิ่ม คลิก https://m.ais.co.th/5Gontop 5G On-top package '
 '55Baht(Inc.VAT). You have got internet 1GB and unlimited internet at speed '
 '6Mbps valid for 24hours. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop ข้อมูลที่เกี่ยวข้อง + AIS 5G Click + '
 'ข้อมูลที่สำคัญเกี่ยวกับ Package ลูกค้าระบบรายเดือน Click + '
 'ข้อมูลที่สำคัญเกี่ยวกับ Package ลูกค้าระบบเติมเงิน\u200b Click + การใช้งาน '
 'Internet  และ การแจ้งเตือน (Data Policy) Click MKT Owner K.Wongrudee '
 'Surarommaneeyasathie <wongruds@ais.co.th>: ชื่อ แพ็กเกจเสริม, เน็ต5G เล่นได้ '
 '1GB หลังจากนั้นใช้ต่อที่ความเร็ว 6 Mbps ใช้ได้ 1 วัน ค่าบริการ 55 บาท '
 'หักแบบรายครั้ง [Prepaid & Postpaid], Concept Package, '
 'แพ็กเกจเสริมอินเทอร์เน็ต 55 บาท + ใช้งานเน็ตที่ความเร็ว 5G/4G เป็นจำนวน 1 GB '
 'หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps + ค่าบริการ 51.40 บาท '
 '(ไม่รวม VAT)  ll  ราคารวม VAT 55 บาท + แพ็กเกจนี้หักค่าบริการแบบรายครั้ง + '
 'ลูกค้าเริ่มใช้งานได้ หลังจากได้รับ SMS ยืนยันการสมัครแพ็กเกจ + ใช้งานได้นาน '
 '1 วัน, กลุ่มลูกค้า, + ลูกค้า AIS บุคคลธรรมดา / ลูกค้านิติบุคคล (ยกเว้นลูกค้า '
 'MVPN) ♦  ระบบรายเดือน ♦  ระบบเติมเงิน One-2-Call (CBS) + '
 'ลูกค้าที่ใช้สมาร์ทโฟนทั้ง 5G/4G, ระยะเวลาสมัคร, ตั้งแต่วันที่ 31 มกราคม 2567 '
 'เป็นต้นไป, ช่องทางการสมัคร, ช่องทาง SLA การสมัคร 1. AIS Call Center 2. USSD '
 '*777*7722# โทรออก (ยกเว้น : ลูกค้ารายเดือน ที่จดทะเบียนในนามนิติบุคคล) 3. '
 'AIS Shop รวม Serenade Club 4. Telewiz 5. Easy app (ROM) เริ่มวันที่ 3 เม.ย. '
 '2567 เป็นต้นไป มีผลทันที ใช้งานได้หลังจากได้รับ SMS ยืนยัน หมายเหตุ : '
 '**ลูกค้านิติบุคคลสามารถรับสิทธิ์สมัครแพ็กเกจเสริมได้ตามช่องทางการสมัครของทางนิติบุคคล**, '
 'ช่องทาง, SLA การสมัคร, 1. AIS Call Center 2. USSD *777*7722# โทรออก (ยกเว้น '
 ': ลูกค้ารายเดือน ที่จดทะเบียนในนามนิติบุคคล) 3. AIS Shop รวม Serenade Club '
 '4. Telewiz 5. Easy app (ROM) เริ่มวันที่ 3 เม.ย. 2567 เป็นต้นไป, มีผลทันที '
 'ใช้งานได้หลังจากได้รับ SMS ยืนยัน, ช่องทางการตรวจสอบ ยอดการใช้งาน, + my AIS '
 '+ บริการ USSD *121# + บริการ IVR Single Number *121, รายละเอียด '
 'แพ็กเกจเสริมเน็ต แบบรายครั้ง Postpaid Product Code P240122840 Promotion Name '
 '5G_Internet In55B 1GB UL 1Day Prepaid Feature Code 3803085 Promotion Name 5G '
 'Internet 55B 1GB UL 1Day ** ห้ามพนักงาน Callcenter สมัครโดยใช้ Code ROM '
 'ข้อมูลเพื่อทราบ กรณีสมัครแพ็กเกจผ่าน ROM เท่านั้น Feature Code 3803169 '
 'Promotion Name 5G Internet 55B 1GB UL 1Day_ROM ประเภท ใช้งานอินเทอร์เน็ต '
 '5G/4G ค่าบริการ (บาท) ใช้งาน (วัน) On-top Package (One Time Charge) '
 'ใช้งานเน็ตที่ความเร็ว 5G/4G เป็นจำนวน 1 GB '
 'หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps 55 บาท (รวม VAT) 1 วัน ( '
 'Time to Time ) 51.40 บาท (ไม่รวม VAT), รายละเอียด แพ็กเกจเสริมเน็ต '
 'แบบรายครั้ง, Postpaid Product Code P240122840 Promotion Name 5G_Internet '
 'In55B 1GB UL 1Day, Postpaid, Product Code, P240122840, Promotion Name, '
 '5G_Internet In55B 1GB UL 1Day, Prepaid Feature Code 3803085 Promotion Name '
 '5G Internet 55B 1GB UL 1Day ** ห้ามพนักงาน Callcenter สมัครโดยใช้ Code ROM '
 'ข้อมูลเพื่อทราบ กรณีสมัครแพ็กเกจผ่าน ROM เท่านั้น Feature Code 3803169 '
 'Promotion Name 5G Internet 55B 1GB UL 1Day_ROM, Prepaid, Feature Code, '
 '3803085, Promotion Name, 5G Internet 55B 1GB UL 1Day, Feature Code, 3803169, '
 'Promotion Name, 5G Internet 55B 1GB UL 1Day_ROM, ประเภท, ใช้งานอินเทอร์เน็ต '
 '5G/4G, ค่าบริการ (บาท), ใช้งาน (วัน), On-top Package (One Time Charge), '
 'ใช้งานเน็ตที่ความเร็ว 5G/4G เป็นจำนวน 1 GB '
 'หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps, 55 บาท (รวม VAT), 1 วัน ( '
 'Time to Time ), 51.40 บาท (ไม่รวม VAT), เงื่อนไขการใช้งาน, '
 'เงื่อนไขแพ็กเกจเสริม 1. อัตราค่าบริการที่กำหนดเป็นอัตรารวมภาษีมูลค่าเพิ่ม 2. '
 'สิทธิ์การใช้งานแพ็กเกจเสริม สามารถใช้งานได้หลังจากได้รับ SMS ยืนยัน 3. '
 'สำหรับผู้ใช้สมาร์ทโฟนที่รองรับ 5G พร้อมทั้งอยู่ในพื้นที่ให้บริการ 5G '
 'สิทธิ์การใช้งานแพ็กเสริมเน็ต ใช้ได้ทั้ง 5G/4G '
 'โดยความเร็วสูงสุดของแพ็กเกจเป็น 5G/4G 4. สำหรับผู้ใช้สมาร์ทโฟนที่ '
 'ไม่รับรองรับ 5G สามารถสมัครแพ็กเกจเสริมนี้ได้ '
 'โดยสิทธิ์การใช้งานแพ็กเสริมเน็ต ใช้ได้ทั้ง 4G '
 'โดยความเร็วสูงสุดของแพ็กเกจเป็น 4G 5. '
 'อัตราค่าบริการอินเทอร์เน็ตคิดตามการใช้งานเป็นกิโลไบท์ (KB) '
 'ส่วนเกินคิดค่าบริการตามแพ็กเกจหลัก 6. ผู้ใช้บริการ MultiSIM '
 'ไม่สามารถสมัครแพ็กเกจอินเทอร์เน็ตไม่จำกัด '
 'หรือแพ็กเกจอินเทอร์เน็ตที่มีการใช้งานต่อเนื่องได้ 7. '
 'สิทธิ์การใช้งานและอัตราค่าบริการที่กำหนด สำหรับการใช้งานภายในประเทศ '
 'และการใช้งานปกติเท่านั้น 8. เพื่อเป็นการรักษามาตรฐานคุณภาพการให้บริการ '
 'และเพื่อให้ผู้ใช้บริการโดยรวมสามารถใช้งานได้ อย่างมีประสิทธิภาพ บริษัทฯ '
 'ขอสงวนสิทธิ์ในการบริหารจัดการเครือข่ายตามความเหมาะสม เช่น '
 'จำกัดหรือลดความเร็วในกรณีที่พบว่ามีการรับส่งข้อมูลในปริมาณมากอย่างต่อเนื่อง '
 'การใช้งาน BitTorrent การแชร์เน็ตผ่าน Hotspot การดาวน์โหลด และ/หรือ '
 'อัปโหลดไฟล์ขนาดใหญ่ หรือใช้งานในเชิงพาณิชย์ '
 'หรือมีการใช้ซึ่งส่งผลกระทบหรือก่อให้เกิดความไม่เป็นธรรมต่อผู้ใช้บริการรายอื่น '
 'หรือกระทบต่อเครือข่ายการให้บริการโดยรวมของบริษัท วิธีการเปิดใช้บริการ AIS 5G '
 'สำหรับผู้ใช้สมาร์ทโฟนที่รองรับ เพียงเปิดบริการ AIS 5G สามารถใช้ AIS 5G '
 'กับแพ็กเกจอินเทอร์เน็ตปัจจุบัน โดยไม่มีค่าใช้จ่ายเพิ่มเติม + '
 'กรณีลูกค้าซื้อสมาร์ทโฟน 5G จาก AIS จะได้รับบริการ AIS 5G อัตโนมัติ + '
 'ลูกค้าที่สมัครแพ็กเกจเสริมสำเร็จ จะได้รับการเปิดบริการ 5G ให้โดยอัตโนมัติ '
 'เงื่อนไขการเปิดบริการ AIS 5G + สงวนสิทธิ์การเปิดบริการ 1 หมายเลข/1 IMEI '
 'เครื่องเท่านั้น ** กรณีหมายเลขหรือเครื่องเคยเปิดบริการไปแล้ว '
 'ไม่สามารถรับสิทธิ์ซ้ำได้อีก **, SMS ยืนยัน การสมัครแพ็กสำเร็จ, Postpaid '
 'แพ็กเกจเสริม 5G ค่าบริการ 55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด '
 'เร็ว 6Mbps ใช้ได้ 24ชม. เริ่ม[Start date] ถึง [End date] ซื้อแพ็กเน็ตเพิ่ม '
 'คลิก https://m.ais.co.th/5Gontop 5G On-top package 55Baht(Inc.VAT). You have '
 'got internet 1GB and unlimited internet at speed 6Mbps valid for 24hours. '
 'Start[Start date] to [End date]. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop Prepaid คุณเริ่มใช้แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'ซื้อแพ็กเน็ตเพิ่ม คลิก https://m.ais.co.th/5Gontop 5G On-top package '
 '55Baht(Inc.VAT). You have got internet 1GB and unlimited internet at speed '
 '6Mbps valid for 24hours. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop, Postpaid, แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'เริ่ม[Start date] ถึง [End date] ซื้อแพ็กเน็ตเพิ่ม คลิก '
 'https://m.ais.co.th/5Gontop, 5G On-top package 55Baht(Inc.VAT). You have got '
 'internet 1GB and unlimited internet at speed 6Mbps valid for 24hours. '
 'Start[Start date] to [End date]. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop, Prepaid, คุณเริ่มใช้แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'ซื้อแพ็กเน็ตเพิ่ม คลิก https://m.ais.co.th/5Gontop, 5G On-top package '
 '55Baht(Inc.VAT). You have got internet 1GB and unlimited internet at speed '
 '6Mbps valid for 24hours. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop, ข้อมูลที่เกี่ยวข้อง, + AIS 5G Click + '
 'ข้อมูลที่สำคัญเกี่ยวกับ Package ลูกค้าระบบรายเดือน Click + '
 'ข้อมูลที่สำคัญเกี่ยวกับ Package ลูกค้าระบบเติมเงิน\u200b Click + การใช้งาน '
 'Internet  และ การแจ้งเตือน (Data Policy) Click, MKT Owner, K.Wongrudee '
 'Surarommaneeyasathie <wongruds@ais.co.th>',
 'ชื่อ แพ็กเกจเสริม: เน็ต5G เล่นได้ 1GB หลังจากนั้นใช้ต่อที่ความเร็ว 6 Mbps '
 'ใช้ได้ 1 วัน ค่าบริการ 55 บาท หักแบบรายครั้ง [Prepaid & Postpaid]',
 'Concept Package: แพ็กเกจเสริมอินเทอร์เน็ต 55 บาท + ใช้งานเน็ตที่ความเร็ว '
 '5G/4G เป็นจำนวน 1 GB หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps + '
 'ค่าบริการ 51.40 บาท (ไม่รวม VAT)  ll  ราคารวม VAT 55 บาท + '
 'แพ็กเกจนี้หักค่าบริการแบบรายครั้ง + ลูกค้าเริ่มใช้งานได้ หลังจากได้รับ SMS '
 'ยืนยันการสมัครแพ็กเกจ + ใช้งานได้นาน 1 วัน',
 'กลุ่มลูกค้า: + ลูกค้า AIS บุคคลธรรมดา / ลูกค้านิติบุคคล (ยกเว้นลูกค้า MVPN) '
 '♦  ระบบรายเดือน ♦  ระบบเติมเงิน One-2-Call (CBS) + ลูกค้าที่ใช้สมาร์ทโฟนทั้ง '
 '5G/4G',
 'ระยะเวลาสมัคร: ตั้งแต่วันที่ 31 มกราคม 2567 เป็นต้นไป',
 'ช่องทางการสมัคร: ช่องทาง SLA การสมัคร 1. AIS Call Center 2. USSD *777*7722# '
 'โทรออก (ยกเว้น : ลูกค้ารายเดือน ที่จดทะเบียนในนามนิติบุคคล) 3. AIS Shop รวม '
 'Serenade Club 4. Telewiz 5. Easy app (ROM) เริ่มวันที่ 3 เม.ย. 2567 '
 'เป็นต้นไป มีผลทันที ใช้งานได้หลังจากได้รับ SMS ยืนยัน หมายเหตุ : '
 '**ลูกค้านิติบุคคลสามารถรับสิทธิ์สมัครแพ็กเกจเสริมได้ตามช่องทางการสมัครของทางนิติบุคคล**, '
 'ช่องทาง, SLA การสมัคร, 1. AIS Call Center 2. USSD *777*7722# โทรออก (ยกเว้น '
 ': ลูกค้ารายเดือน ที่จดทะเบียนในนามนิติบุคคล) 3. AIS Shop รวม Serenade Club '
 '4. Telewiz 5. Easy app (ROM) เริ่มวันที่ 3 เม.ย. 2567 เป็นต้นไป, มีผลทันที '
 'ใช้งานได้หลังจากได้รับ SMS ยืนยัน',
 'ช่องทาง: SLA การสมัคร',
 '1. AIS Call Center 2. USSD *777*7722# โทรออก (ยกเว้น : ลูกค้ารายเดือน '
 'ที่จดทะเบียนในนามนิติบุคคล) 3. AIS Shop รวม Serenade Club 4. Telewiz 5. Easy '
 'app (ROM) เริ่มวันที่ 3 เม.ย. 2567 เป็นต้นไป: มีผลทันที '
 'ใช้งานได้หลังจากได้รับ SMS ยืนยัน',
 'ช่องทางการตรวจสอบ ยอดการใช้งาน: + my AIS + บริการ USSD *121# + บริการ IVR '
 'Single Number *121',
 'รายละเอียด แพ็กเกจเสริมเน็ต แบบรายครั้ง Postpaid Product Code P240122840 '
 'Promotion Name 5G_Internet In55B 1GB UL 1Day Prepaid Feature Code 3803085 '
 'Promotion Name 5G Internet 55B 1GB UL 1Day ** ห้ามพนักงาน Callcenter '
 'สมัครโดยใช้ Code ROM ข้อมูลเพื่อทราบ กรณีสมัครแพ็กเกจผ่าน ROM เท่านั้น '
 'Feature Code 3803169 Promotion Name 5G Internet 55B 1GB UL 1Day_ROM ประเภท '
 'ใช้งานอินเทอร์เน็ต 5G/4G ค่าบริการ (บาท) ใช้งาน (วัน) On-top Package (One '
 'Time Charge) ใช้งานเน็ตที่ความเร็ว 5G/4G เป็นจำนวน 1 GB '
 'หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps 55 บาท (รวม VAT) 1 วัน ( '
 'Time to Time ) 51.40 บาท (ไม่รวม VAT): รายละเอียด แพ็กเกจเสริมเน็ต '
 'แบบรายครั้ง, Postpaid Product Code P240122840 Promotion Name 5G_Internet '
 'In55B 1GB UL 1Day, Postpaid, Product Code, P240122840, Promotion Name, '
 '5G_Internet In55B 1GB UL 1Day, Prepaid Feature Code 3803085 Promotion Name '
 '5G Internet 55B 1GB UL 1Day ** ห้ามพนักงาน Callcenter สมัครโดยใช้ Code ROM '
 'ข้อมูลเพื่อทราบ กรณีสมัครแพ็กเกจผ่าน ROM เท่านั้น Feature Code 3803169 '
 'Promotion Name 5G Internet 55B 1GB UL 1Day_ROM, Prepaid, Feature Code, '
 '3803085, Promotion Name, 5G Internet 55B 1GB UL 1Day, Feature Code, 3803169, '
 'Promotion Name, 5G Internet 55B 1GB UL 1Day_ROM, ประเภท, ใช้งานอินเทอร์เน็ต '
 '5G/4G, ค่าบริการ (บาท), ใช้งาน (วัน), On-top Package (One Time Charge), '
 'ใช้งานเน็ตที่ความเร็ว 5G/4G เป็นจำนวน 1 GB '
 'หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps, 55 บาท (รวม VAT), 1 วัน ( '
 'Time to Time ), 51.40 บาท (ไม่รวม VAT)',
 'Postpaid Product Code P240122840 Promotion Name 5G_Internet In55B 1GB UL '
 '1Day: Postpaid, Product Code, P240122840, Promotion Name, 5G_Internet In55B '
 '1GB UL 1Day, Prepaid Feature Code 3803085 Promotion Name 5G Internet 55B 1GB '
 'UL 1Day ** ห้ามพนักงาน Callcenter สมัครโดยใช้ Code ROM ข้อมูลเพื่อทราบ '
 'กรณีสมัครแพ็กเกจผ่าน ROM เท่านั้น Feature Code 3803169 Promotion Name 5G '
 'Internet 55B 1GB UL 1Day_ROM, Prepaid, Feature Code, 3803085, Promotion '
 'Name, 5G Internet 55B 1GB UL 1Day, Feature Code, 3803169, Promotion Name, 5G '
 'Internet 55B 1GB UL 1Day_ROM',
 'Product Code: P240122840',
 'Promotion Name: 5G Internet 55B 1GB UL 1Day_ROM',
 'Feature Code: 3803169',
 'ประเภท: ใช้งานอินเทอร์เน็ต 5G/4G, ค่าบริการ (บาท), ใช้งาน (วัน)',
 'On-top Package (One Time Charge): ใช้งานเน็ตที่ความเร็ว 5G/4G เป็นจำนวน 1 GB '
 'หลังจากนั้นใช้ได้ต่อเนื่องที่ความเร็วที่ 6 Mbps, 55 บาท (รวม VAT), 1 วัน ( '
 'Time to Time )',
 'เงื่อนไขการใช้งาน: เงื่อนไขแพ็กเกจเสริม 1. '
 'อัตราค่าบริการที่กำหนดเป็นอัตรารวมภาษีมูลค่าเพิ่ม 2. '
 'สิทธิ์การใช้งานแพ็กเกจเสริม สามารถใช้งานได้หลังจากได้รับ SMS ยืนยัน 3. '
 'สำหรับผู้ใช้สมาร์ทโฟนที่รองรับ 5G พร้อมทั้งอยู่ในพื้นที่ให้บริการ 5G '
 'สิทธิ์การใช้งานแพ็กเสริมเน็ต ใช้ได้ทั้ง 5G/4G '
 'โดยความเร็วสูงสุดของแพ็กเกจเป็น 5G/4G 4. สำหรับผู้ใช้สมาร์ทโฟนที่ '
 'ไม่รับรองรับ 5G สามารถสมัครแพ็กเกจเสริมนี้ได้ '
 'โดยสิทธิ์การใช้งานแพ็กเสริมเน็ต ใช้ได้ทั้ง 4G '
 'โดยความเร็วสูงสุดของแพ็กเกจเป็น 4G 5. '
 'อัตราค่าบริการอินเทอร์เน็ตคิดตามการใช้งานเป็นกิโลไบท์ (KB) '
 'ส่วนเกินคิดค่าบริการตามแพ็กเกจหลัก 6. ผู้ใช้บริการ MultiSIM '
 'ไม่สามารถสมัครแพ็กเกจอินเทอร์เน็ตไม่จำกัด '
 'หรือแพ็กเกจอินเทอร์เน็ตที่มีการใช้งานต่อเนื่องได้ 7. '
 'สิทธิ์การใช้งานและอัตราค่าบริการที่กำหนด สำหรับการใช้งานภายในประเทศ '
 'และการใช้งานปกติเท่านั้น 8. เพื่อเป็นการรักษามาตรฐานคุณภาพการให้บริการ '
 'และเพื่อให้ผู้ใช้บริการโดยรวมสามารถใช้งานได้ อย่างมีประสิทธิภาพ บริษัทฯ '
 'ขอสงวนสิทธิ์ในการบริหารจัดการเครือข่ายตามความเหมาะสม เช่น '
 'จำกัดหรือลดความเร็วในกรณีที่พบว่ามีการรับส่งข้อมูลในปริมาณมากอย่างต่อเนื่อง '
 'การใช้งาน BitTorrent การแชร์เน็ตผ่าน Hotspot การดาวน์โหลด และ/หรือ '
 'อัปโหลดไฟล์ขนาดใหญ่ หรือใช้งานในเชิงพาณิชย์ '
 'หรือมีการใช้ซึ่งส่งผลกระทบหรือก่อให้เกิดความไม่เป็นธรรมต่อผู้ใช้บริการรายอื่น '
 'หรือกระทบต่อเครือข่ายการให้บริการโดยรวมของบริษัท วิธีการเปิดใช้บริการ AIS 5G '
 'สำหรับผู้ใช้สมาร์ทโฟนที่รองรับ เพียงเปิดบริการ AIS 5G สามารถใช้ AIS 5G '
 'กับแพ็กเกจอินเทอร์เน็ตปัจจุบัน โดยไม่มีค่าใช้จ่ายเพิ่มเติม + '
 'กรณีลูกค้าซื้อสมาร์ทโฟน 5G จาก AIS จะได้รับบริการ AIS 5G อัตโนมัติ + '
 'ลูกค้าที่สมัครแพ็กเกจเสริมสำเร็จ จะได้รับการเปิดบริการ 5G ให้โดยอัตโนมัติ '
 'เงื่อนไขการเปิดบริการ AIS 5G + สงวนสิทธิ์การเปิดบริการ 1 หมายเลข/1 IMEI '
 'เครื่องเท่านั้น ** กรณีหมายเลขหรือเครื่องเคยเปิดบริการไปแล้ว '
 'ไม่สามารถรับสิทธิ์ซ้ำได้อีก **',
 'SMS ยืนยัน การสมัครแพ็กสำเร็จ: Postpaid แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'เริ่ม[Start date] ถึง [End date] ซื้อแพ็กเน็ตเพิ่ม คลิก '
 'https://m.ais.co.th/5Gontop 5G On-top package 55Baht(Inc.VAT). You have got '
 'internet 1GB and unlimited internet at speed 6Mbps valid for 24hours. '
 'Start[Start date] to [End date]. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop Prepaid คุณเริ่มใช้แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'ซื้อแพ็กเน็ตเพิ่ม คลิก https://m.ais.co.th/5Gontop 5G On-top package '
 '55Baht(Inc.VAT). You have got internet 1GB and unlimited internet at speed '
 '6Mbps valid for 24hours. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop, Postpaid, แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'เริ่ม[Start date] ถึง [End date] ซื้อแพ็กเน็ตเพิ่ม คลิก '
 'https://m.ais.co.th/5Gontop, 5G On-top package 55Baht(Inc.VAT). You have got '
 'internet 1GB and unlimited internet at speed 6Mbps valid for 24hours. '
 'Start[Start date] to [End date]. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop, Prepaid, คุณเริ่มใช้แพ็กเกจเสริม 5G ค่าบริการ '
 '55บาท(รวมVAT) อินเทอร์เน็ต 1GB พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. '
 'ซื้อแพ็กเน็ตเพิ่ม คลิก https://m.ais.co.th/5Gontop, 5G On-top package '
 '55Baht(Inc.VAT). You have got internet 1GB and unlimited internet at speed '
 '6Mbps valid for 24hours. To enjoy more internet, click '
 'https://m.ais.co.th/5Gontop',
 'Postpaid: แพ็กเกจเสริม 5G ค่าบริการ 55บาท(รวมVAT) อินเทอร์เน็ต 1GB '
 'พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. เริ่ม[Start date] ถึง [End date] '
 'ซื้อแพ็กเน็ตเพิ่ม คลิก https://m.ais.co.th/5Gontop',
 'Prepaid: คุณเริ่มใช้แพ็กเกจเสริม 5G ค่าบริการ 55บาท(รวมVAT) อินเทอร์เน็ต 1GB '
 'พร้อมเน็ตไม่จำกัด เร็ว 6Mbps ใช้ได้ 24ชม. ซื้อแพ็กเน็ตเพิ่ม คลิก '
 'https://m.ais.co.th/5Gontop',
 'ข้อมูลที่เกี่ยวข้อง: + AIS 5G Click + ข้อมูลที่สำคัญเกี่ยวกับ Package '
 'ลูกค้าระบบรายเดือน Click + ข้อมูลที่สำคัญเกี่ยวกับ Package '
 'ลูกค้าระบบเติมเงิน\u200b Click + การใช้งาน Internet  และ การแจ้งเตือน (Data '
 'Policy) Click',
 'MKT Owner: K.Wongrudee Surarommaneeyasathie <wongruds@ais.co.th>']
# !!! END OF EDITING SECTION !!!


@st.cache_resource
def load_models():
    """Loads the embedding and language models."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return embedding_model, llm_tokenizer, llm_model

embedding_model, llm_tokenizer, llm_model = load_models()

try:
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
except Exception as e:
    st.error(f"Could not load the FAISS index file '{FAISS_INDEX_PATH}'. Make sure the file is in the same folder as app.py.")
    st.stop()

def answer_question(question, top_k=3):
    """Finds relevant docs and generates an answer using an LLM."""
    question_embedding = embedding_model.encode([question])
    distances, indices = faiss_index.search(np.array(question_embedding).astype('float32'), top_k)

    retrieved_docs = [documents[i] for i in indices[0]]
    context = " ".join(retrieved_docs)
 
    prompt = f"ในฐานะผู้ช่วยบริการลูกค้าของ AIS จงตอบคำถามนี้อย่างสุภาพเป็นประโยคที่สมบูรณ์: '{question}' โดยใช้ข้อมูลนี้เท่านั้น: '{context}'"
 

    input_ids = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids
    outputs = llm_model.generate(input_ids, max_length=256, num_beams=5, early_stopping=True)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, context

# --- Streamlit User Interface ---
st.title("🤖 Chatbot ถาม-ตอบข้อมูลโปรโมชั่น")
st.write("ป้อนคำถามเกี่ยวกับโปรโมชั่น แล้ว Chatbot จะค้นหาข้อมูลและตอบคำถามให้คุณ")

user_question = st.text_input("คำถามของคุณ:")

if st.button("ส่งคำถาม"):
    if user_question:
        with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
            answer, context = answer_question(user_question)
            st.subheader("คำตอบ:")
            st.write(answer)
            with st.expander("ข้อมูลที่ใช้ในการตอบ (Context)"):
                st.write(context)
    else:
        st.warning("กรุณาป้อนคำถาม")
