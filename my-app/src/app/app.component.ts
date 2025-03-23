import { Component } from '@angular/core';
import { AppService } from './app.service';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  selectedFile: File;
  prediction: string | null = null;
  imageUrl: string | null = null;

  title = 'my-app';
  imageURL: string; // Variable to store the image URL

  constructor(private appService: AppService, private http: HttpClient, private sanitizer: DomSanitizer) {}

  images: { id: number, src: string, alt: string }[] = [
    { id: 1, src: 'https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/627d124572023b6948b6cdff_60ed9a4e09e2c648f1b8a013_object-detection-cover.png', alt: 'Image 1' },
    { id: 2, src: 'https://blog.roboflow.com/content/images/2024/04/image-1056.webp', alt: 'Image 2' },
    { id: 3, src: 'https://viso.ai/wp-content/uploads/2021/02/9-1-people-detection-meeting-room-1060x596.jpg', alt: 'Image 3' },
    { id: 4, src: 'https://learn.g2.com/hubfs/G2CM_FI264_Learn_Article_Images_%5BObject_detection%5D_V1a.png', alt: 'Image 4' },
    { id: 5, src: 'https://www.datasciencecentral.com/wp-content/uploads/2021/10/9712908078.jpeg', alt: 'Image 5' }
    // Add more images as needed
  ];

  onFileSelected(event) {
    this.selectedFile = event.target.files[0];
  }

  onUpload() {
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('file', this.selectedFile);

      this.http.post<any>('http://localhost:5000/predict', formData).subscribe(
        response => {
          console.log('Prediction:', response);
          this.prediction = response.predicted_class;
          // Assuming response.imageUrl is the URL of the predicted image
          this.imageUrl = response.image_base64;

          // Assuming 'response' contains the JSON response from the backend
          const predictedClass = response.predicted_class;
          const imgBase64 = response.image_base64;

          // Display the predicted class
          document.getElementById('predicted-class').textContent = predictedClass;

          // Display the image
          const imgElement = document.getElementById('predicted-image') as HTMLImageElement;
          imgElement.src = 'data:image/jpeg;base64,' + imgBase64;
        },
        error => {
          console.error('Error:', error);
        }
      );
    } else {
      console.error('No file selected');
    }
  }

  // Sanitize the image URL
  getImageUrl() {
    return this.sanitizer.bypassSecurityTrustUrl(this.imageUrl);
  }  
}
