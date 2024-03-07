task1:
	Part1: data collection
	先將load.py中第12的test_scene改成你儲存apartment_0/habitat/mesh_semantic.ply的位置
	執行load.py
	先用w,s,a,d移動到所要的區域,按下p鍵同時儲存BEV view 與 front view image，資料會被儲存於images_for_projection資料夾,
	接著按下f鍵結束load.py.
	
	Part2: BEV Projection
	先執行bev.py
	會跳出一張front view image, 對想要的位置按下滑鼠左鍵, 全部完成後按任意鍵便會自度投影到BEV view, 並顯示出來，
	最後按下任意鍵結束bev.py.
	
task2:
	Part1: data collection
	先執行load.py
	可以使用r鍵儲存資料，分別為rgb_img(存於images資料夾), dapth_img(存於images資料夾)和 position(存於posit_data.txt)，
	並用wsad移動到所要位置，在此建議一開始先儲存一次資料，接下來最多每隔兩步(w,s,a,d個算一步)就儲存資料，以使ICP
	疊圖不會出錯，存完一圈資料後可以按f鍵結束load.py.

	Part2:  Point Cloud Alignment and Reconstruction
	執行reconstuct.py
	此程式會自動將2D資料轉成3D(pcd形式)，並將每次各個pcd資料存於pcd_data資料夾內，因此會花較久的時間
	接著分會跑出三張圖，分別為重建後的pcd、加上estimate的pcd與加上estimate與ground truth 的pcd。
	第一章圖為Part2所要之重建圖
task3:
	於剛剛步驟所顯示之第二張與第三張變為estimate與groundtruth的pcd，
	而L1 distance會於終端機顯示。
		