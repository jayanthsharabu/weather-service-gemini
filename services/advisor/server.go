package advisor

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
	"weatherservices/shared/proto/advisorpb"
	"weatherservices/shared/proto/weatherpb"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"google.golang.org/api/option"

	"github.com/google/generative-ai-go/genai"
)

var (
	advisorRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "advisor_requests_total",
			Help: "Total advisor requests",
		},
		[]string{"status"},
	)
	advisorDuration = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name: "advisor_request_duration_seconds",
			Help: "Advisor request duration",
		},
	)
)

type advisorService struct {
	advisorpb.UnimplementedAdvisorServiceServer
	weatherSvc  *weatherpb.WeatherServiceServer
	genaiClient *genai.Client
}

type geoCodeResponse struct {
	Results []struct {
		Name      string  `json: "name"`
		Latitude  float64 `json: "latitude"`
		Longitude float64 `json: "longitude"`
		Country   string  `json: "country"`
		Admin1    string  `json: "admin1"`
	} `json: "results"`
}

func NewAdvisorService(weatherSvc *weatherpb.WeatherServiceServer, geminiAPIkey string) (*advisorService, error) {
	ctx := context.Background()
	genaiClient, err := genai.NewClient(ctx, option.WithAPIKey(geminiAPIkey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %v", err)
	}
	return &advisorService{
		weatherSvc:  weatherSvc,
		genaiClient: genaiClient,
	}, nil
}

func (s *advisorService) Close() {
	if s.genaiClient != nil {
		s.genaiClient.Close()
	}
}

func (s *advisorService) geocodeCity(_ context.Context, city *advisorpb.CityData) (float64, float64, error) {
	encodedQuery := url.QueryEscape(city.Location)
	url := fmt.Sprintf("https://nominatim.openstreetmap.org/search?q=%s&format=json&limit=1", encodedQuery)
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return 0, 0, fmt.Errorf("geocoding failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("geocoding failed: %s", resp.Status)
	}
	var geoResponse geoCodeResponse
	if err := json.NewDecoder(resp.Body).Decode(&geoResponse); err != nil {
		return 0, 0, fmt.Errorf("JSON decoding failed: %v", err)
	}
	if len(geoResponse.Results) == 0 {
		return 0, 0, fmt.Errorf("no results found for city: %s", city.Location)
	}
	return geoResponse.Results[0].Latitude, geoResponse.Results[0].Longitude, nil
}

func (s *advisorService) getAdvice(ctx context.Context, advisorRequest *advisorpb.AdvisorRequest) (*advisorpb.AdvisorResponse, error) {
	timer := prometheus.NewTimer(advisorDuration)
	defer timer.ObserveDuration()

	var weatherData []string
	for _, city := range advisorRequest.Cities {
		latitude, longitude, err := s.geocodeCity(ctx, city)
		if err != nil {
			advisorRequests.WithLabelValues("error").Inc()
			return nil, fmt.Errorf("geocoding failed for %s: %v", city.Location, err)
		}

		weatherReq := &weatherpb.WeatherRequest{Latitude: latitude, Longitude: longitude}
		weatherResp, err := (*s.weatherSvc).GetCurrentWeather(ctx, weatherReq)
		if err != nil {
			advisorRequests.WithLabelValues("error").Inc()
			return nil, fmt.Errorf("weather request failed for %s: %v", city.Location, err)
		}
		weatherInfo := fmt.Sprintf("City: %s, Temp: %.1f°C, Condition: %s, Humidity: %d%%, Wind: %.1f m/s",
			city.Location, weatherResp.Temperature, weatherResp.Description, weatherResp.Humidity, weatherResp.WindSpeed)
		weatherData = append(weatherData, weatherInfo)
	}

	advice, err := s.generateAdvice(ctx, weatherData)
	if err != nil {
		advisorRequests.WithLabelValues("error").Inc()
		return nil, fmt.Errorf("advice generation failed: %v", err)
	}
	advisorRequests.WithLabelValues("success").Inc()
	return &advisorpb.AdvisorResponse{Advice: advice}, nil
}

func (s *advisorService) generateAdvice(ctx context.Context, weatherData []string) (string, error) {
	model := s.genaiClient.GenerativeModel("gemini-2.5-flash")
	prompt := fmt.Sprintf(`Weather advisor. Based on this data provide practical advice: %s Include: summary, clothing advice, activity suggestions, places to visit if good weather, warnings. Keep it concise.`, strings.Join(weatherData, "\n"))
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("gemini API failed: %v", err)
	}

	if len(resp.Candidates) == 0 {
		return "", fmt.Errorf("no response generated")
	}

	var advice strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			advice.WriteString(string(text))
		}
	}
	return advice.String(), nil

}

func (s *advisorService) StreamAdvice(req *advisorpb.AdvisorRequest, stream advisorpb.AdvisorService_StreamAdviceServer) error {
	timer := prometheus.NewTimer(advisorDuration)
	defer timer.ObserveDuration()

	var weatherData, failedCities []string

	for _, city := range req.Cities {
		latitude, longitude, err := s.geocodeCity(stream.Context(), city)
		if err != nil {
			failedCities = append(failedCities, city.Location)
			continue
		}
		weatherReq := &weatherpb.WeatherRequest{Latitude: latitude, Longitude: longitude}
		weatherResp, err := (*s.weatherSvc).GetCurrentWeather(stream.Context(), weatherReq)
		if err != nil {
			failedCities = append(failedCities, fmt.Sprintf("%s (weather failed)", city.Location))
			continue
		}

		weatherInfo := fmt.Sprintf("City: %s, Temp: %.1f°C, Condition: %s, Humidity: %d%%, Wind: %.1f m/s",
			city.Location, weatherResp.Temperature, weatherResp.Description, weatherResp.Humidity, weatherResp.WindSpeed)
		weatherData = append(weatherData, weatherInfo)
	}
	if len(weatherData) == 0 {
		message := "i could not get any weather data for any of the cities"
		if len(failedCities) > 0 {
			message += fmt.Sprintf("\nFailed to get weather data for: %s", strings.Join(failedCities, ", "))
		}
		message += "\nPlease check the city names and try again."

		err := stream.Send(&advisorpb.StreamAdviceResponse{
			Chunk:      message,
			IsComplete: true,
		})
		if err == nil {
			advisorRequests.WithLabelValues("success").Inc()
		} else {
			advisorRequests.WithLabelValues("error").Inc()
		}
		return err
	}

	err := s.streamAdviceGeneration(stream.Context(), weatherData, stream)
	if err != nil {
		advisorRequests.WithLabelValues("error").Inc()
		return fmt.Errorf("advice generation failed: %v", err)
	}

	advisorRequests.WithLabelValues("success").Inc()
	return nil
}

func (s *advisorService) streamAdviceGeneration(ctx context.Context, weatherData []string, stream advisorpb.AdvisorService_StreamAdviceServer) error {
	model := s.genaiClient.GenerativeModel("gemini-2.5-pro")
	prompt := fmt.Sprintf(`Weather advisor. Based on this data provide practical advice: %s Include: summary, clothing advice, activity suggestions, warnings. Keep it concise.`, strings.Join(weatherData, "\n"))

	iter := model.GenerateContentStream(ctx, genai.Text(prompt))

	for {
		resp, err := iter.Next()
		if err != nil {
			if strings.Contains(err.Error(), "EOF") || strings.Contains(err.Error(), "iterator stopped") {

				return stream.Send(&advisorpb.StreamAdviceResponse{
					Chunk:      "",
					IsComplete: true,
				})
			}
			return fmt.Errorf("streaming failed: %v", err)
		}

		for _, cand := range resp.Candidates {
			for _, part := range cand.Content.Parts {
				if text, ok := part.(genai.Text); ok {
					err := stream.Send(&advisorpb.StreamAdviceResponse{
						Chunk:      string(text),
						IsComplete: false,
					})
					if err != nil {
						return fmt.Errorf("failed to send chunk: %v", err)
					}
				}
			}
		}
	}
}
