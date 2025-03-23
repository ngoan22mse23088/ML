import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AppService {
  private baseUrl = 'http://127.0.0.1:5000'; // Thay thế bằng URL của backend của bạn

  constructor(private http: HttpClient) { }

  // Phương thức để gửi yêu cầu POST đến backend
  postData(image: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', image);

    return this.http.post<any>(`${this.baseUrl}/predict`, formData);
  }
  // postData(image: File): Observable<any> {
  //   const formData = new FormData();
  //   formData.append('file', image);

  //   // Các tùy chọn header có thể thay đổi tùy theo backend của bạn
  //   const httpOptions = {
  //     headers: new HttpHeaders({
  //       'Content-Type': 'multipart/form-data'
  //     })
  //   };

  //   return this.http.post<any>(`${this.baseUrl}/predict`, formData, httpOptions);
  // }
}
